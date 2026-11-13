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
    for (const model of TTS_MODELS) {
        try {
            // Gemini 3.8 TTS czyta tekst dosłownie — nie dopisujemy żadnych poleceń w treści
            const r = await ai.models.generateContent({
                model,
                contents: [{ parts: [{ text }] }],
                config: {
                    responseModalities: ['AUDIO'],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: TTS_VOICE } } }
                }
            });
            const b64 = r.candidates?.[0]?.content?.parts?.find(p => p.inlineData)?.inlineData?.data;
            if (!b64) continue;
            const buf = Buffer.from(b64, 'base64');
            // Nowe modele zwracają gotowy WAV (nagłówek RIFF), starsze surowe PCM
            const out = buf.slice(0, 4).toString('ascii') === 'RIFF' ? buf : pcmToWav(buf);
            return out.toString('base64');
        } catch (e) {
            lastErr = e;
            console.error(`TTS (${model}) nie zadziałał:`, e.message);
        }
    }
    if (lastErr) throw lastErr;
    return null;
}

app.get('/', (req, res) => res.send('OK'));

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
- "textResponse": pełny tekst analizy,
- "audioPrompt": wersja pod syntezator mowy: czysty tekst bez markdownu, wypunktowań, symboli i emoji, krótkie zdania, liczby zapisane tak, by dobrze brzmiały na głos, maksymalnie około 250 słów.
`;

        // UWAGA: googleSearch nie działa razem z responseMimeType: 'application/json' — JSON parsujemy ręcznie
        const response = await ai.models.generateContent({
            model: TEXT_MODEL,
            contents: prompt,
            config: noData ? {} : { tools: [{ googleSearch: {} }] }
        });

        const raw = response.text; // w @google/genai to getter, nie funkcja
        const parsed = parseJson(raw) || { textResponse: raw, audioPrompt: raw };
        const textResponse = parsed.textResponse || raw;
        const audioPrompt = parsed.audioPrompt || textResponse;

        let audio = null;
        try {
            audio = await makeAudio(audioPrompt);
        } catch (e) {
            console.error('TTS nie zadziałał, widget użyje głosu przeglądarki:', e.message);
        }

        const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
        res.json({
            success: true,
            analysis: textResponse,
            audioPrompt,
            audio, // base64 WAV albo null
            sources: chunks.map(c => c.web?.uri).filter(Boolean)
        });
    } catch (error) {
        console.error('Błąd podczas analizy:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
