const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors());

// Klucz pobierany ze zmiennej środowiskowej GEMINI_API_KEY (Render → Environment)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const TEXT_MODEL = 'gemini-2.5-flash';
const TTS_MODEL = 'gemini-2.5-flash-preview-tts';
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
    const r = await ai.models.generateContent({
        model: TTS_MODEL,
        contents: [{ parts: [{ text: 'Przeczytaj po polsku, spokojnym i naturalnym tonem, jak kolega analityk: ' + text }] }],
        config: {
            responseModalities: ['AUDIO'],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: TTS_VOICE } } }
        }
    });
    const b64 = r.candidates?.[0]?.content?.parts?.find(p => p.inlineData)?.inlineData?.data;
    if (!b64) return null;
    return pcmToWav(Buffer.from(b64, 'base64')).toString('base64');
}

app.get('/', (req, res) => res.send('OK'));

app.post('/api/analyze', async (req, res) => {
    try {
        const { sessionsHistory = [], currentSession = null, channelName = 'Dolaricious' } = req.body;

        const prompt = `
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
            config: { tools: [{ googleSearch: {} }] }
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
