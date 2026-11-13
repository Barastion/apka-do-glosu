const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors());

// Klucz pobierany ze zmiennej środowiskowej GEMINI_API_KEY (Render → Environment)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Modele można zmienić w Render → Environment bez ruszania kodu (TEXT_MODEL / TTS_MODEL)
const TEXT_MODEL = process.env.TEXT_MODEL || 'gemini-3.8-flash';
const TTS_MODELS = [process.env.TTS_MODEL || 'gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts'];
const TTS_VOICE = process.env.TTS_VOICE || 'Kore';
const TEXT_MODELS = [...new Set([TEXT_MODEL, process.env.TEXT_MODEL_FALLBACK || 'gemini-3.7-flash'])];

const withTimeout = (p, ms, label) => Promise.race([
    p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label}: przekroczono limit ${ms / 1000}s`)), ms))
]);

// Wywołanie modelu tekstowego: najpierw z niskim poziomem „myślenia” (szybciej), a gdyby model odrzucił ten parametr — bez niego
async function gen(prompt, search, ms, model = TEXT_MODEL) {
    const base = search ? { tools: [{ googleSearch: {} }] } : {};
    const tries = [{ ...base, thinkingConfig: { thinkingLevel: 'low' } }, base];
    let last = null;
    for (const config of tries) {
        try {
            return await withTimeout(ai.models.generateContent({ model, contents: prompt, config }), ms, `Analiza (${model})`);
        } catch (e) {
            last = e;
            if (/przekroczono limit/.test(e.message)) throw e; // timeout → nie ponawiamy tu, robi to wywołujący
            console.error('Próba generowania nieudana:', e.message);
        }
    }
    throw last;
}

// Zbiera adresy źródeł z odpowiedzi Interactions API (adnotacje url_citation)
function collectSources(i) {
    const out = new Set();
    for (const st of i?.steps || []) for (const c of st.content || []) for (const a of c.annotations || []) if (a?.url) out.add(a.url);
    return [...out];
}

// Jedno podejście do modelu: najpierw Interactions API (zalecane przez Google), potem klasyczne generateContent
async function genOne(prompt, search, ms, model) {
    try {
        const req = { model, input: prompt, generation_config: { thinking_level: 'low' } };
        if (search) req.tools = [{ type: 'google_search' }];
        const i = await withTimeout(ai.interactions.create(req), ms, `Analiza (${model})`);
        if (!i.output_text) throw new Error('pusta odpowiedź');
        return { text: i.output_text, model, via: 'interactions', sources: collectSources(i) };
    } catch (e) {
        if (/przekroczono limit/.test(e.message)) throw e;
        console.error(`Interactions (${model}) nie wyszło:`, e.message);
    }
    const r = await gen(prompt, search, ms, model);
    const chunks = r.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    return { text: r.text, model, via: 'generateContent', sources: chunks.map(c => c.web?.uri).filter(Boolean) };
}

// Wszystkie modele naraz — bierzemy pierwszą udaną odpowiedź (gdy jeden jest przeciążony, drugi ratuje sytuację)
async function genRace(prompt, search, ms) {
    const t0 = Date.now();
    try {
        const r = await Promise.any(TEXT_MODELS.map(m => genOne(prompt, search, ms, m)));
        console.log(`Analiza OK (${r.model}, ${r.via}${search ? ', z wyszukiwaniem' : ''}): ${Date.now() - t0} ms`);
        return r;
    } catch (e) {
        const msgs = (e.errors || [e]).map(x => x.message).join(' | ');
        throw new Error(msgs);
    }
}

function parseJson(t) {
    const c = String(t || '').replace(/```json|```/gi, '').trim();
    try { return JSON.parse(c); } catch (e) {}
    const m = c.match(/\{[\s\S]*\}/);
    if (m) { try { return JSON.parse(m[0]); } catch (e) {} }
    return null;
}

// Gemini TTS zwraca surowe PCM 16-bit / 24 kHz / mono — pakujemy w WAV, żeby przeglądarka mogła odtworzyć
function pcmToWav(pcm, sampleRate = 24000, channels = 1, bits = 16) {
    const h = Buffer.alloc(44);
    h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVE', 8);
    h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(channels, 22);
    h.writeUInt32LE(sampleRate, 24); h.writeUInt32LE(sampleRate * channels * bits / 8, 28);
    h.writeUInt16LE(channels * bits / 8, 32); h.writeUInt16LE(bits, 34);
    h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
    return Buffer.concat([h, pcm]);
}

// Wyciąga audio (base64) z odpowiedzi Interactions API
function pickAudio(i) {
    if (i?.output_audio?.data) return i.output_audio.data;
    const steps = i?.steps || i?.outputs || [];
    for (const st of [...steps].reverse()) {
        for (const c of [...(st.content || [])].reverse()) if (c?.type === 'audio' && c.data) return c.data;
    }
    return null;
}

// Gemini 3.8 TTS działa przez Interactions API (generateContent z speechConfig jest przez Google odrzucane):
// tekst w input, głos w generation_config.speech_config, format w response_format
async function makeAudio(text, models = TTS_MODELS) {
    let lastErr = null;
    const styles = [null, 'calm, natural, friendly, conversational pace']; // najpierw bez stylu (zalecane przez Google)
    for (const model of models) {
        for (const style of styles) {
            try {
                const part = { type: 'text', text };
                if (style) part.annotations = [{ type: 'speech_metadata', style }];
                const interaction = await withTimeout(ai.interactions.create({
                    model,
                    input: [{ type: 'user_input', content: [part] }],
                    response_format: { type: 'audio' },
                    generation_config: { speech_config: [{ voice: TTS_VOICE }] }
                }), 30000, `TTS ${model}`);
                const b64 = pickAudio(interaction);
                if (!b64) { lastErr = new Error(`TTS ${model}: brak audio w odpowiedzi`); continue; }
                const buf = Buffer.from(b64, 'base64');
                // unary zwraca gotowy WAV (RIFF); gdyby przyszedł surowy PCM — dopinamy nagłówek
                const out = buf.slice(0, 4).toString('ascii') === 'RIFF' ? buf : pcmToWav(buf);
                console.log(`TTS OK: ${model}${style ? ' (ze stylem)' : ''}`);
                return out.toString('base64');
            } catch (e) {
                lastErr = e;
                console.error(`TTS (${model}${style ? ', ze stylem' : ''}) nie zadziałał:`, e.message);
                if (/przekroczono limit/.test(e.message)) break; // timeout → następny model
            }
        }
    }
    if (lastErr) throw lastErr;
    return null;
}

app.get('/', (req, res) => res.send('OK'));
app.get('/health', (req, res) => res.json({ ok: true, textModel: TEXT_MODEL, ttsModels: TTS_MODELS }));

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Diagnostyka: mierzy czas każdego modelu (tekst + głos) — otwórz w przeglądarce /diag
app.get('/diag', async (req, res) => {
    const timed = async (label, fn) => {
        const t0 = Date.now();
        try { await fn(); return { label, ok: true, ms: Date.now() - t0 }; }
        catch (e) { return { label, ok: false, ms: Date.now() - t0, error: e.message }; }
    };
    const out = await Promise.all([
        ...TEXT_MODELS.flatMap(m => [
            timed('tekst (interactions) ' + m, () => withTimeout(ai.interactions.create({
                model: m, input: 'Odpowiedz jednym słowem: ok', generation_config: { thinking_level: 'low' }
            }), 45000, 'tekst')),
            timed('tekst (generateContent) ' + m, () => withTimeout(ai.models.generateContent({
                model: m, contents: 'Odpowiedz jednym słowem: ok', config: { thinkingConfig: { thinkingLevel: 'low' } }
            }), 45000, 'tekst'))
        ]),
        ...TTS_MODELS.map(m => timed('głos ' + m, () => makeAudio('Cześć, tu Gemini. Test głosu.', [m])))
    ]);
    res.json(out);
});

// Strona testowa głosu: /tts-test  (opcjonalnie ?model=gemini-3.8-flash-lite-tts&text=...)
app.get('/tts-test', async (req, res) => {
    const text = String(req.query.text || 'Cześć! Tu Gemini. Sprawdzamy, czy mój głos działa poprawnie i brzmi naturalnie po polsku.').slice(0, 600);
    const models = req.query.model ? [String(req.query.model)] : TTS_MODELS;
    const t0 = Date.now();
    try {
        const audio = await makeAudio(text, models);
        if (!audio) throw new Error('Brak audio w odpowiedzi');
        res.send(`<meta charset="utf-8"><body style="font:16px sans-serif;background:#111;color:#eee;padding:20px"><p>Model: ${esc(models.join(' → '))} · czas: ${Date.now() - t0} ms · głos: ${esc(TTS_VOICE)}</p><audio controls autoplay src="data:audio/wav;base64,${audio}"></audio><p>${esc(text)}</p></body>`);
    } catch (e) {
        res.status(500).send(`<meta charset="utf-8"><body style="font:16px sans-serif;background:#111;color:#eee;padding:20px"><p>BŁĄD po ${Date.now() - t0} ms</p><pre style="white-space:pre-wrap;color:#ff8">${esc(e.message)}</pre></body>`);
    }
});

// Osobny endpoint głosu — widżet woła go po otrzymaniu tekstu
app.post('/api/tts', async (req, res) => {
    try {
        const text = String(req.body?.text || '').slice(0, 1500);
        if (!text) return res.status(400).json({ success: false, error: 'Brak tekstu' });
        const audio = await makeAudio(text);
        if (!audio) throw new Error('Model nie zwrócił audio');
        res.json({ success: true, audio });
    } catch (error) {
        console.error('Błąd TTS:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.post('/api/analyze', async (req, res) => {
    try {
        const { sessionsHistory = [], currentSession = null, channelName = 'Dolaricious', noData = false } = req.body;

        const emptyPrompt = `
Jesteś asystentem-analitykiem kanału Twitch streamera ${channelName}. Widżet nie ma jeszcze zapisanej żadnej zakończonej sesji, więc nie ma czego analizować.
Napisz po polsku krótką, życzliwą wypowiedź (ok. 70 słów): przywitaj się, powiedz wprost, że na razie brak danych z sesji, i że po pierwszej zakończonej transmisji omówisz wyniki oraz szerszy obraz polskiego Twitcha. Potwierdź, że głos działa poprawnie. Nie podawaj żadnych statystyk ani liczb.
Zwróć WYŁĄCZNIE JSON: {"textResponse": "...", "audioPrompt": "..."} — audioPrompt to ten sam tekst, czysty, bez markdownu i emoji.
`;

        const prompt = noData ? emptyPrompt : `
Jesteś profesjonalnym analitykiem kanału Twitch oraz polskiego rynku streamingowego dla streamera o nicku ${channelName}.

Dane historycznych sesji z widgetu (najnowsze pierwsze):
${JSON.stringify(sessionsHistory, null, 2)}

Dane najnowszej sesji (pole "trwaJeszcze" mówi, czy sesja jeszcze trwa):
${JSON.stringify(currentSession, null, 2)}

Uwaga: pola zaczynające się od "szac_" to szacunki widgetu, a nie twarde dane z Twitcha.

Przeanalizuj te dane i przygotuj wypowiedź w języku polskim:
1. Około 50% wypowiedzi: konkretne wyniki najnowszej sesji (co zadziałało, co nie, subskrypcje, followy, aktywność czatu, porównanie z wcześniejszymi sesjami). Nie używaj nicków widzów.
2. Pozostała część: szerszy obraz (długoterminowe wzorce, gry, pory, dni tygodnia, trendy polskiego Twitcha oraz okazje/nadchodzące wydarzenia).
3. Bezwzględnie odróżniaj twarde dane z historii od szacunków i prognoz rynkowych. Jeśli publiczne źródła nie pozwalają precyzyjnie określić liczby polskich widzów w kategorii, powiedz to wprost i nie zmyślaj statystyk.
4. Zwięzła, partnerska forma (jak inteligentny kolega), nastawiona na wartość poznawczą i strategiczną.

Zwróć WYŁĄCZNIE obiekt JSON (bez markdownu, bez komentarzy) z polami:
- "textResponse": pełny tekst analizy (maksymalnie około 300 słów),
- "audioPrompt": wersja pod syntezator mowy: czysty tekst bez markdownu, wypunktowań, symboli i emoji, krótkie zdania, liczby zapisane tak, by dobrze brzmiały na głos, maksymalnie około 200 słów.
`;

        // Brak danych → nie potrzeba modelu tekstowego: gotowa wypowiedź idzie prosto do głosu (test działa nawet gdy tekstowy Gemini jest przeciążony)
        if (noData) {
            const txt = `Cześć! Tu Twój analityk kanału ${channelName}. Na razie nie mam zapisanych żadnych zakończonych sesji, więc nie ma jeszcze czego analizować. Po pierwszej zakończonej transmisji omówię wyniki, a potem szerszy obraz polskiego Twitcha. Ten głos pochodzi z Gemini, więc wszystko działa poprawnie.`;
            return res.json({ success: true, analysis: txt, audioPrompt: txt, searchUsed: false, sources: [] });
        }

        // UWAGA: wyszukiwanie Google nie działa razem z wymuszonym JSON — JSON parsujemy ręcznie
        let result = null, searchUsed = false, lastErr = null;
        try {
            result = await genRace(prompt, true, 40000);
            searchUsed = true;
        } catch (e) { lastErr = e; console.error('Analiza z wyszukiwaniem nie wyszła:', e.message); }
        if (!result) {
            const fallbackPrompt = prompt + '\n\nUWAGA: wyszukiwarka jest teraz niedostępna. Opieraj się wyłącznie na przesłanych danych sesji, a w części o rynku wprost powiedz, że bieżących danych o polskim Twitchu nie sprawdzałeś.';
            try { result = await genRace(fallbackPrompt, false, 35000); }
            catch (e) { lastErr = e; console.error('Analiza bez wyszukiwania nie wyszła:', e.message); }
        }
        if (!result) throw lastErr || new Error('Analiza nie powiodła się');

        const raw = result.text;
        const parsed = parseJson(raw) || { textResponse: raw, audioPrompt: raw };
        const textResponse = parsed.textResponse || raw;
        const audioPrompt = parsed.audioPrompt || textResponse;

        res.json({ success: true, analysis: textResponse, audioPrompt, searchUsed, sources: result.sources || [] });
    } catch (error) {
        console.error('Błąd podczas analizy:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
