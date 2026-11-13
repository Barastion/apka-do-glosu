const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors());

// Inicjalizacja Gemini SDK (klucz pobierany ze zmiennych środowiskowych Render.com)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post('/api/analyze', async (req, res) => {
    try {
        const { sessionsHistory, currentSession, channelName = "Dolaricious" } = req.body;

        // Prompt systemowy wymuszający podział na 50% ostatnia sesja / 50% szerszy obraz oraz twarde dane vs szacunki
        const prompt = `
Jesteś profesjonalnym analitykiem kanału Twitch oraz polskiego rynku streamingowego dla streamera o nicku ${channelName}.
Oto dane historyczne sesji przesłane z widgetu:
${JSON.stringify(sessionsHistory, null, 2)}

Oto dane z ostatniej zakończonej sesji:
${JSON.stringify(currentSession, null, 2)}

Przeanalizuj te dane i przygotuj wypowiedź w języku polskim, zachowując następującą strukturę:
1. Około 50% czasu wypowiedzi: konkretne wyniki najnowszej sesji (co zadziałało, co nie, subskrypcje, followy, aktywność czatu, porównanie z wcześniejszymi wynikami). Nie używaj nicków widzów.
2. Pozostała część: szerszy obraz (długoterminowe wzorce, gry, pory, dni tygodnia, trendy polskiego Twitcha oraz okazje/nadchodzące wydarzenia).
3. Bezwzględnie odróżniaj twarde dane z historii od szacunków i prognoz rynkowych. Jeśli publiczne źródła lub dane nie pozwalają precyzyjnie określić liczby polskich widzów w kategorii, zaznacz to wprost, nie zmyślaj statystyk.
4. Odpowiedz w zwięzłej, partnerskiej formie (jak inteligentny kolega/analityk), nastawionej na wartość poznawczą i strategiczną.

Zwróć wynik wyłącznie w formacie JSON z następującymi polami:
- "textResponse": pełny tekst analizy (gotowy do wyświetlenia w okienku),
- "audioPrompt": wersja tekstu zoptymalizowana pod syntezator mowy (TTS).
`;

        // Wywołanie Gemini z włączonym wyszukiwaniem Google (Search Grounding) do analizy rynku
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: {
                tools: [{ googleSearch: {} }],
                responseMimeType: 'application/json'
            }
        });

        const resultText = response.text();
        const parsedData = JSON.parse(resultText);

        res.json({
            success: true,
            analysis: parsedData.textResponse,
            audioPrompt: parsedData.audioPrompt,
            sources: response.candidates?.[0]?.groundingMetadata?.webSearchQueries || []
        });

    } catch (error) {
        console.error("Błąd podczas analizy:", error);
        res.status(500).json({ success: false, error: error.message });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
    console.log(`Serwer analityczny działa na porcie ${PORT}`);
});
