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
const TTS_VOICE = 'Kore';

const withTimeout = (p, ms, label) => Promise.race([
    p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label}: przekroczono limit ${ms / 1000}s`)), ms))
]);

// Wywołanie modelu tekstowego: najpierw z niskim poziomem „myślenia” (szybciej), a gdyby model odrzucił ten parametr — bez niego
async function gen(prompt, search, ms) {
    const base = search ? { tools: [{ googleSearch: {} }] } : {};
    const tries = [{ ...base, thinkingConfig: { thinkingLevel: 'low' } }, base];
    let last = null;
    for (const config of tries) {
        try {
            return await withTimeout(ai.models.generateContent({ model: TEXT_MODEL, contents: prompt, config }), ms, 'Analiza');
        } catch (e) {
            last = e;
            if (/przekroczono limit/.test(e.message)) throw e; // timeout → nie ponawiamy tu, robi to wywołujący
            console.error('Próba generowania nieudana:', e.message);
        }
    }
    throw last;
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

async function makeAudio(text) {
    let lastErr = null;
    // Schemat Gemini 3.8 TTS: parts[].text = dosłowny transkrypt, styl w speech_metadata, głos w voiceConfig.voice.
    // Na wypadek zmian po stronie Google próbujemy kilku wariantów (z/bez stylu, nowy/stary zapis głosu).
    const variants = [
        { part: { text, speech_metadata: { style: 'spokojny, naturalny, przyjazny ton polskiego analityka, w tempie rozmowy' } }, voiceConfig: { voice: TTS_VOICE } },
        { part: { text }, voiceConfig: { voice: TTS_VOICE } },
        { part: { text }, voiceConfig: { prebuiltVoiceConfig: { voiceName: TTS_VOICE } } }
    ];
    for (const model of TTS_MODELS) {
        for (const v of variants) {
            try {
                const r = await withTimeout(ai.models.generateContent({
                    model,
                    contents: [{ role: 'user', parts: [v.part] }],
                    config: { responseModalities: ['AUDIO'], speechConfig: { voiceConfig: v.voiceConfig } }
                }), 40000, `TTS ${model}`);
                const b64 = r.candidates?.[0]?.content?.parts?.find(p => p.inlineData)?.inlineData?.data;
                if (!b64) { lastErr = new Error(`TTS ${model}: brak audio w odpowiedzi`); continue; }
                const buf = Buffer.from(b64, 'base64');
                // Nowe modele zwracają gotowy WAV (nagłówek RIFF), starsze surowe PCM
                const out = buf.slice(0, 4).toString('ascii') === 'RIFF' ? buf : pcmToWav(buf);
                console.log(`TTS OK: ${model}, wariant ${variants.indexOf(v) + 1}`);
                return out.toString('base64');
            } catch (e) {
                lastErr = e;
                console.error(`TTS (${model}, wariant ${variants.indexOf(v) + 1}) nie zadziałał:`, e.message);
                if (/przekroczono limit/.test(e.message)) break; // timeout → następny model
            }
        }
    }
    if (lastErr) throw lastErr;
    return null;
}

app.get('/', (req, res) => res.send('OK'));
app.get('/health', (req, res) => res.json({ ok: true, textModel: TEXT_MODEL, ttsModels: TTS_MODELS }));

// Osobny endpoint głosu — widżet woła go po otrzymaniu tekstu
app.post('/api/tts', async (req, res) => {
    try {
        const text = String(req.body?.text || '').slice(0, 4000);
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

        // UWAGA: googleSearch nie działa razem z responseMimeType: 'application/json' — JSON parsujemy ręcznie
        // Najpierw z wyszukiwaniem Google (do 45 s); jeśli trwa zbyt długo lub zawiedzie — szybka wersja bez wyszukiwania
        let response = null, searchUsed = false;
        if (!noData) {
            const t0 = Date.now();
            try {
                response = await gen(prompt, true, 45000);
                searchUsed = true;
                console.log(`Analiza z wyszukiwaniem: ${Date.now() - t0} ms`);
            } catch (e) {
                console.error('Analiza z wyszukiwaniem nie wyszła, próbuję bez niego:', e.message);
            }
        }
        if (!response) {
            const t1 = Date.now();
            const fallbackPrompt = noData ? prompt : prompt + '\n\nUWAGA: wyszukiwarka jest teraz niedostępna. Opieraj się wyłącznie na przesłanych danych sesji, a w części o rynku wprost powiedz, że bieżących danych o polskim Twitchu nie sprawdzałeś.';
            response = await gen(fallbackPrompt, false, 40000);
            console.log(`Analiza bez wyszukiwania: ${Date.now() - t1} ms`);
        }

        const raw = response.text; // w @google/genai to getter, nie funkcja
        const parsed = parseJson(raw) || { textResponse: raw, audioPrompt: raw };
        const textResponse = parsed.textResponse || raw;
        const audioPrompt = parsed.audioPrompt || textResponse;

        const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
        res.json({
            success: true,
            analysis: textResponse,
            audioPrompt,
            searchUsed,
            sources: chunks.map(c => c.web?.uri).filter(Boolean)
        });
    } catch (error) {
        console.error('Błąd podczas analizy:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
