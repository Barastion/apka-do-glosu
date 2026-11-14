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

// Poziom „myślenia”: na żywo szybko, offline głębiej (zmienne środowiskowe THINK_LIVE / THINK_OFFLINE)
const THINK = { live: process.env.THINK_LIVE || 'low', offline: process.env.THINK_OFFLINE || 'medium' };
// Limity czasu analizy: na żywo ok. minuty, offline maks. ok. 2 minuty (z zapasem na wyszukiwanie)
const LIMITS = {
    live:    { search: 50000, fallback: 25000 },
    offline: { search: 90000, fallback: 25000 }
};

// Style głosu: na żywo — radiowy, lekko komentatorski; offline — analityczny (głos nadal kobiecy: Kore)
const TTS_STYLES = {
    live: 'warm, energetic female radio presenter, lightly sports-commentator style, engaging and upbeat, lively pace; the final call-to-action sentences are soft, friendly and unobtrusive',
    offline: 'calm, analytical, professional female voice, clear and measured pace, natural and friendly'
};

const withTimeout = (p, ms, label) => Promise.race([
    p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label}: przekroczono limit ${ms / 1000}s`)), ms))
]);

// Wywołanie modelu tekstowego: najpierw z zadanym poziomem „myślenia”, a gdyby model odrzucił ten parametr — bez niego
async function gen(prompt, search, ms, model = TEXT_MODEL, think = 'low') {
    const base = search ? { tools: [{ googleSearch: {} }] } : {};
    const tries = [{ ...base, thinkingConfig: { thinkingLevel: think } }, base];
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
async function genOne(prompt, search, ms, model, think) {
    try {
        const req = { model, input: prompt, generation_config: { thinking_level: think } };
        if (search) req.tools = [{ type: 'google_search' }];
        const i = await withTimeout(ai.interactions.create(req), ms, `Analiza (${model})`);
        if (!i.output_text) throw new Error('pusta odpowiedź');
        return { text: i.output_text, model, via: 'interactions', sources: collectSources(i) };
    } catch (e) {
        if (/przekroczono limit/.test(e.message)) throw e;
        console.error(`Interactions (${model}) nie wyszło:`, e.message);
    }
    const r = await gen(prompt, search, ms, model, think);
    const chunks = r.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    return { text: r.text, model, via: 'generateContent', sources: chunks.map(c => c.web?.uri).filter(Boolean) };
}

// Wszystkie modele naraz — bierzemy pierwszą udaną odpowiedź (gdy jeden jest przeciążony, drugi ratuje sytuację)
async function genRace(prompt, search, ms, think) {
    const t0 = Date.now();
    try {
        const r = await Promise.any(TEXT_MODELS.map(m => genOne(prompt, search, ms, m, think)));
        console.log(`Analiza OK (${r.model}, ${r.via}${search ? ', z wyszukiwaniem' : ''}, think=${think}): ${Date.now() - t0} ms`);
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
    const ma = c.match(/"audioPrompt"\s*:\s*"([\s\S]*?)"\s*[,}]/), mt = c.match(/"textResponse"\s*:\s*"([\s\S]*?)"\s*[,}]/);
    if (ma || mt) { const un = x => x.replace(/\\n/g, ' ').replace(/\\"/g, '"'); return { audioPrompt: ma ? un(ma[1]) : undefined, textResponse: mt ? un(mt[1]) : undefined }; }
    return null;
}

// ═══════════════ DATA I CZAS (jedyne, co liczy serwer; święta/wydarzenia sprawdza sam model w sieci) ═══════════════
const TZ = 'Europe/Warsaw';
function warsawParts(d) {
    const f = new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'long', hour: '2-digit', minute: '2-digit', hour12: false });
    return Object.fromEntries(f.formatToParts(d).map(x => [x.type, x.value]));
}
const warsawYmd = d => { const p = warsawParts(d); return `${p.year}-${p.month}-${p.day}`; };
const dayDiff = (a, b) => Math.round((Date.parse(warsawYmd(a) + 'T00:00:00Z') - Date.parse(warsawYmd(b) + 'T00:00:00Z')) / 864e5); // a − b w dniach (czas warszawski)
const clock = d => new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, hour: '2-digit', minute: '2-digit', hour12: false }).format(d);

function calendarContext(now = new Date()) {
    const p = warsawParts(now), dow = new Date(Date.UTC(+p.year, +p.month - 1, +p.day)).getUTCDay();
    return {
        dzis: {
            data: new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, day: 'numeric', month: 'long', year: 'numeric' }).format(now),
            dzienTygodnia: p.weekday, godzina: `${p.hour}:${p.minute}`, dataISO: `${p.year}-${p.month}-${p.day}`, weekend: dow === 0 || dow === 6
        }
    };
}

// Dopisuje do osi czasu sesji godzinę zegarową (minuty od startu → HH:MM), żeby model widział, o której rosła i spadała widownia
function withClock(s) {
    if (!s || !Array.isArray(s.przebieg) || !s.start) return s;
    const t0 = Date.parse(s.start);
    return { ...s, przebieg: s.przebieg.map(x => ({ godzina: clock(new Date(t0 + (x.min || 0) * 6e4)), ...x })) };
}

// ═══════════════ WYLICZENIA STATYSTYCZNE (robimy w kodzie, model tylko interpretuje) ═══════════════
const round = (n, d = 0) => Number.isFinite(n) ? Math.round(n * 10 ** d) / 10 ** d : null;
const pct = (a, b) => (Number.isFinite(a) && Number.isFinite(b) && b > 0) ? Math.round((a - b) / b * 100) : null;
const band = hh => hh >= 6 && hh < 12 ? 'rano (6–11)' : hh >= 12 && hh < 17 ? 'popołudnie (12–16)' : hh >= 17 && hh < 22 ? 'wieczór (17–21)' : 'noc (22–5)';

function sessionMetrics(s) {
    const h = Math.max(0.05, (s.czasTrwaniaMin || 0) / 60);
    return {
        h,
        avg: s.sredniaWidzow || 0,
        peak: s.peakWidzow || 0,
        msgH: (s.wiadomosciNaCzacie || 0) / h,
        folH: (s.nowiFollowersi || 0) / h,
        subH: (s.noweSuby || 0) / h,
        chatShare: s.szac_unikalniWidzowie > 0 ? (s.unikalniCzatujacy || 0) / s.szac_unikalniWidzowie : null,
        msgPerViewerHour: (s.sredniaWidzow > 0) ? (s.wiadomosciNaCzacie || 0) / (s.sredniaWidzow * h) : null
    };
}
function wavg(items, key) { // średnia ważona (wagi malejące dla starszych sesji)
    let ws = 0, vs = 0;
    for (const it of items) { const v = it.m[key]; if (v == null || !Number.isFinite(v)) continue; ws += it.w; vs += v * it.w; }
    return ws ? vs / ws : null;
}
function groupBy(list, keyFn) {
    const map = {};
    list.forEach(it => { const k = keyFn(it); if (!k) return; (map[k] = map[k] || []).push(it); });
    return Object.entries(map).map(([k, arr]) => ({
        klucz: k, liczbaSesji: arr.length, lacznieGodzin: round(arr.reduce((a, x) => a + x.m.h, 0), 1),
        sredniaWidzow_wazona: round(wavg(arr, 'avg'), 1), sredniPeak_wazony: round(wavg(arr, 'peak'), 1),
        wiadomosciNaGodzine_wazone: round(wavg(arr, 'msgH'), 1), nowiFollowersiNaGodzine_wazone: round(wavg(arr, 'folH'), 2),
        noweSubyNaGodzine_wazone: round(wavg(arr, 'subH'), 2), wagaOstatnichSesji: round(arr.reduce((a, x) => a + x.w, 0), 2)
    })).sort((a, b) => (b.sredniaWidzow_wazona || 0) - (a.sredniaWidzow_wazona || 0));
}

function derive(current, history) {
    const all = [...(current ? [current] : []), ...(history || [])].filter(s => s && s.czasTrwaniaMin > 0 && s.start);
    // wagi: najnowsza sesja = 1, każda starsza × 0.82 — ostatnie gry i terminy liczą się wyraźnie bardziej
    const items = all.map((s, i) => ({ s, m: sessionMetrics(s), w: Math.pow(0.82, i) }));
    const histItems = items.filter(it => it.s !== current);
    const out = { liczbaSesjiWHistorii: histItems.length };
    out.wedlugGier = groupBy(items, it => it.s.glownaGra);
    out.wedlugDniTygodnia = groupBy(items, it => it.s.dzienTygodnia);
    out.wedlugPorStartu = groupBy(items, it => { const hh = parseInt(String(it.s.godzinaStartu || '').slice(0, 2), 10); return Number.isFinite(hh) ? band(hh) : null; });
    out.wedlugGodzinyStartu = groupBy(items, it => { const hh = parseInt(String(it.s.godzinaStartu || '').slice(0, 2), 10); return Number.isFinite(hh) ? String(hh).padStart(2, '0') + ':00' : null; });
    if (current && histItems.length) {
        const cm = sessionMetrics(current);
        const base = k => wavg(histItems, k);
        const mk = (arr) => arr.length ? {
            sredniaWidzow: round(wavg(arr, 'avg'), 1), peak: round(wavg(arr, 'peak'), 1), wiadomosciNaGodzine: round(wavg(arr, 'msgH'), 1),
            nowiFollowersiNaGodzine: round(wavg(arr, 'folH'), 2), liczbaSesjiWBazie: arr.length
        } : null;
        out.porownanieNajnowszejSesji = {
            uwaga: 'Porównanie po znormalizowaniu do godziny streama (wiadomości i followersi na godzinę); średnia widzów i peak są porównywane bezpośrednio, ale pamiętaj o różnicy długości sesji.',
            czasTrwaniaNajnowszejH: round(cm.h, 2), sredniCzasTrwaniaHistoriiH: round(base('h'), 2),
            odchylenieProcent_wzgledemWazonejHistorii: {
                sredniaWidzow: pct(cm.avg, base('avg')), peak: pct(cm.peak, base('peak')),
                wiadomosciNaGodzine: pct(cm.msgH, base('msgH')), nowiFollowersiNaGodzine: pct(cm.folH, base('folH'))
            },
            historiaWazona: mk(histItems),
            tenSamDzienTygodnia: mk(histItems.filter(it => it.s.dzienTygodnia === current.dzienTygodnia)),
            taSamaGra: current.glownaGra ? mk(histItems.filter(it => it.s.glownaGra === current.glownaGra)) : null,
            takaSamaPoraStartu: mk(histItems.filter(it => band(parseInt(String(it.s.godzinaStartu || '0').slice(0, 2), 10)) === band(parseInt(String(current.godzinaStartu || '0').slice(0, 2), 10)))),
            udzialCzatujacychWSzacowanychUnikalnych: { najnowsza: round(cm.chatShare, 2), historia: round(base('chatShare'), 2) },
            wiadomosciNaWidzaNaGodzine: { najnowsza: round(cm.msgPerViewerHour, 2), historia: round(base('msgPerViewerHour'), 2) }
        };
    }
    return out;
}

function lastSessionNote(current, cal, live) {
    if (!current) return 'Brak zapisanej historii sesji w widżecie.';
    const startD = new Date(current.start), now = new Date();
    if (live) {
        const mins = Math.max(0, Math.round((now - startD) / 6e4));
        return `Stream trwa teraz. Start: ${current.dzienTygodnia}, godz. ${current.godzinaStartu} (${mins} min temu).`;
    }
    const endD = current.koniec ? new Date(current.koniec) : new Date(startD.getTime() + (current.czasTrwaniaMin || 0) * 6e4);
    const dd = dayDiff(now, endD), hoursAgo = Math.round((now - endD) / 36e5);
    const end = clock(endD);
    if (dayDiff(now, startD) === 0 || dd === 0) return `Dziś był już stream: start ${current.godzinaStartu}, koniec ok. ${end}, czyli ${hoursAgo} godz. temu.`;
    return `Dziś nie było jeszcze streama. Ostatni miał miejsce ${dd === 1 ? 'wczoraj' : dd + ' dni temu'} (${current.dzienTygodnia}, start ${current.godzinaStartu}, koniec ok. ${end}).`;
}

// ═══════════════ PROMPT ═══════════════
function buildPrompt(body, cal, derived) {
    const { channelName = 'Dolaricious', sessionsHistory = [], currentSession = null, live = false, kanal = {}, teraz = {}, wariantCTA = 'obserwowanie' } = body;
    const note = lastSessionNote(currentSession, cal, live);
    const ctaTxt = wariantCTA === 'obserwowanie_i_suby'
        ? 'subtelnie zachęć widzów do zaobserwowania kanału ORAZ do zostawienia suba (jedno–dwa miękkie zdania)'
        : 'subtelnie zachęć widzów wyłącznie do zaobserwowania kanału (jedno miękkie zdanie, bez wzmianki o subskrypcji)';

    const modeBlock = live ? `
TRYB: STREAM TRWA (analiza „na żywo”). Prawdopodobnie widzowie też Cię słyszą.
- Skup się na bieżącej sesji i oceniaj bieżące wydarzenia na tle historii.
- Obowiązkowo wspomnij: aktualną liczbę widzów, średnią sesji, liczbę wiadomości na czacie i liczbę piszących, nowe followy i suby w tej sesji (jeśli są), o której zaczął się stream, dzień tygodnia, w jaką grę i pod jakim tytułem trwa transmisja.
- Odczytaj z słupków (pole "aktualnie": słupki widzów i czatu oraz ich trendy), czy trend widowni i czatu jest rosnący, malejący czy stabilny.
- Sprawdź w sieci, ile osób w Polsce ogląda teraz tę grę i ilu streamerów ją streamuje (jeśli się da), oraz największe wydarzenia na Twitchu dziś.
- Styl: radiowy, lekko komentatorski, angażujący widzów, kobiecy, ale rzeczowy. Krótkie, rytmiczne zdania.
- Zakończ lekkim CTA: ${ctaTxt}. Ma być nienachalne, naturalne i bez błagania.
- Długość: tekst do wypowiedzenia ok. 150–180 słów (ok. minuty mówienia). Wyszukaj w sieci maksymalnie 4–5 razy (zaplanuj zapytania tak, by jedno obejmowało kalendarz i wydarzenia, a pozostałe grę i Twitcha), żeby zmieścić się w czasie.` : `
TRYB: STREAM ZAKOŃCZONY / OFFLINE.
- Ok. 50% wypowiedzi poświęć najnowszej zakończonej sesji: jak poszło, porównanie z historią (z uwzględnieniem czasu trwania), powody odchyleń.
- Jeśli danych o sesjach jest mało, skup się bardziej na danych kanału poza sesją: liczba followersów, tempo przyrostu w ostatnich 30 dniach, suby, daty najnowszych followów i subów.
- Pozostałe 50%: wzorce (najlepsze godziny i dni tygodnia, gry, przyczyny), rynek polski, kalendarz, patche/premiery/wydarzenia, stosunek widzów do streamujących dla gier, w które gram.
- Styl: bardziej analityczny, spokojny, rzeczowy. Bez CTA do widzów.
- Długość: tekst do wypowiedzenia ok. 280–320 słów (do 2 minut mówienia). Możesz wyszukać w sieci do ok. 10 razy.`;

    return `
Jesteś analitykiem kanału Twitch streamera ${channelName} (polski rynek streamingowy). Poniżej masz wszystkie dane z widżetu, wyliczenia wykonane w kodzie oraz dzisiejszą datę. Kalendarz (święta, dni wolne, wydarzenia) sprawdzasz sam w sieci.

═══ ZASADY BEZWZGLĘDNE ═══
1. Nigdy nie czytaj ani nie wymieniaj nicków i nazw użytkowników (poza nazwą ${channelName}). Nigdy nie mów o zarobkach, kwotach, pieniądzach ani przychodach z subskrypcji, bitsów i podobnych. Liczby followersów i subów wolno podawać.
2. Absolutny zakaz kryptoreklamy: nie wymieniaj nazw firm, portali, serwisów, narzędzi, sklepów, marek ani produktów. Wyjątki: tytuły gier, nazwy studiów i wydawców gier oraz sam Twitch. Źródła opisuj ogólnie (np. „publiczne statystyki Twitcha”), bez nazw stron.
3. Nigdy nie doradzaj streamerowi zmiany gry ani przeniesienia streamowania na inną porę lub dzień (np. „graj wieczorem zamiast rano”). WOLNO natomiast i warto: zauważać zmiany aktywności w trakcie jego streama (np. wzrost widowni około 12:00, spadek około 14:00) i podawać ich prawdopodobne przyczyny (koniec pracy lub dojazd, przerwa obiadowa, lekcje, początek weekendu itp.), a także informować o stosunku streamujących do widzów i o wydarzeniach — jako obserwacje („dane pokazują…”), bez poleceń zmiany.
4. Prawda, bez słodzenia i bez różowych okularów, ale też bez pesymizmu: szukaj plusów, wspominaj o minusach. Nie zmyślaj liczb. Odróżniaj twarde dane z historii, szacunki (pola zaczynające się od „szac_”) i prognozy/hipotezy. Jeśli czegoś nie da się sprawdzić, powiedz to wprost.
5. Świeżość: dziś jest ${cal.dzis.dzienTygodnia}, ${cal.dzis.data}, godz. ${cal.dzis.godzina} (czas polski). Wszystko, co sprawdzasz w sieci, weryfikuj na DZIŚ. Patche, aktualizacje, premiery, zmiany wersji (np. beta→pełna wersja), eventy: wspominaj tylko te, które dzieją się dziś lub w przyszłości (podaj kiedy). Zdarzenia z przeszłości dozwolone tylko do 7 dni wstecz i tylko jeśli realnie wpływają na Twitcha; starsze pomijaj. Jeśli nie potwierdzisz informacji na dziś — nie wspominaj o niej.
   KALENDARZA NIE MASZ PODANEGO — sam go sprawdź w sieci na podstawie dzisiejszej daty (zob. punkt I).
6. Dopasuj analizę do polskich widzów i polskiego Twitcha. Statystyki światowe tylko wtedy, gdy są naprawdę istotne.
7. Zawsze rozpocznij od przedstawienia się jako analityk kanału ${channelName}, powiedz jaki jest dziś dzień i data oraz godzina. Następnie: ${note}

═══ CO MASZ ANALIZOWAĆ ═══
A. Wzorce: w jakie gry grał, w jakie dni tygodnia, o której zaczynał i kończył. Wskaż najlepsze godziny i dni (użyj gotowych wyliczeń z "wyliczeniaSerwera", nie licz w pamięci) i szukaj PRZYCZYN: np. piątkowy wieczór — początek weekendu; święto lub dzień wolny od pracy; wakacje albo ferie; przedświąteczne przygotowania; pora roku (weekend w wakacje wygląda inaczej niż weekend zimą); duże wydarzenia, które mogły wciągnąć polską widownię (zob. punkt I).
A2. Wzorce WEWNĄTRZ streama: w osi czasu ("przebieg", każdy punkt ma "godzina" zegarową) oraz w słupkach z "aktualnie" znajdź momenty wzrostu i spadku widowni i czatu, podaj o której godzinie to nastąpiło i wyjaśnij prawdopodobną przyczynę (koniec pracy, dojazd, obiad, lekcje, początek weekendu, rajd, dropy, zmiana gry lub tytułu).
B. Gry: ostatnio streamowane gry mają większą wagę niż stare (waga już policzona w danych) — poświęć im więcej uwagi. Dla tych gier sprawdź w sieci oglądalność w Polsce i w czołówce Twitcha, stosunek liczby widzów do liczby streamujących (mało streamujących, duże zapotrzebowanie = korzystny stosunek) oraz w jakie dni i godziny ten stosunek bywa najlepszy — podaj to wyłącznie jako informację (zasada 3). Sprawdź też patchnote’y, aktualizacje, premiery i duże wydarzenia dotyczące tych gier (zasada 5). Porównaj oglądalność tej gry z topowymi grami w Polsce. Poinformuj też, jaka inna duża gra lub wydarzenie na Twitchu może teraz ściągać widzów (bez sugerowania zmian).
C. Tytuł i gra a wyniki: jeśli w tytule jest np. „DROP”/„drops”, zastanów się, czy dropy mogły przyciągnąć widzów; sprawdź w "przebieg" i polach czatu, czy wzrost widowni przełożył się na piszących (unikalni czatujący, wiadomości). Jeśli widzów przybyło, a czat nie drgnął — to widzowie głównie „po nagrody”; jeśli rosły oba — stream faktycznie dobrze trzyma. Rozróżniaj te sytuacje.
D. Korelacje: więcej widzów + więcej wiadomości = dobry znak; mało widzów, ale obecnych przez większość streama = stabilna publiczność; mało widzów i pisze ta sama grupa osób (wysoki udział czatujących względem unikalnych widzów) = własne grono odbiorców. Wskaż, która z tych sytuacji wygląda na prawdziwą.
E. Rajdy: pole "skokiWidzow" wskazuje nagłe wzrosty z szybkim odpływem. Jeśli taki wzorzec jest, powiedz, że to prawdopodobnie rajd lub hosting (jako hipoteza, nie fakt), że widownia szybko odpłynęła i jak wygląda widownia „bazowa” bez tego skoku. Nie podawaj nicków.
F. Czas trwania: nie porównuj sesji 1:1 — dłuższy stream daje inne sumy niż krótszy. Używaj wartości znormalizowanych (na godzinę) z "porownanieNajnowszejSesji" i wspominaj o różnicy długości.
G. Tempo rozwoju: ile followersów przybyło w ostatnich 30 dniach i jaki to wzrost procentowy, ile saldo po odliczeniu unfollowów, jak wyglądają suby (nowe i wygasłe), kiedy wpadł ostatni follow i ostatni sub. Zauważ i nazwij dobry okres, jeśli suby/followersi rosną. Zaznacz, że część danych subów jest przybliżona, jeśli widżet działa krótko.
H. Za każdym razem szukaj PRZYCZYN odchyleń, a nie samego „o 5% więcej niż zwykle”: lepszy stream, lepsza godzina, dzień wolny, gra, wydarzenie zewnętrzne, rajd, dropy.
I. KALENDARZ I WYDARZENIA — sprawdź SAM w sieci, na podstawie dzisiejszej daty (nic nie jest wgrane): czy dziś lub w najbliższych dniach jest święto państwowe, dzień wolny od pracy, długi weekend lub „most”; czy to weekend w wakacje, w ferie (ferie zimowe sprawdź z podziałem na województwa, jeśli to ich okres), w zimie czy poza sezonem; czy trwają przygotowania świąteczne lub inny okres obniżonej oglądalności. Sprawdź też wydarzenia, które w Polsce masowo odciągają widzów z Twitcha: matury i egzaminy, mecze reprezentacji Polski, duże gale freak-fightowe, finały dużych turniejów e-sportowych (np. IEM Katowice) oraz inne ważne premiery i wydarzenia. Wspominaj wyłącznie to, co potwierdziłeś dla dzisiejszej daty (zasada 5), i wyjaśnij, jak mogło to wpłynąć na widownię.
${modeBlock}

═══ DANE ═══
DZISIEJSZA DATA (z zegara serwera, czas polski) — święta, dni wolne i wydarzenia sprawdzasz sam w sieci:
${JSON.stringify(cal, null, 2)}

CZAS I STAN TERAZ (z widżetu):
${JSON.stringify(teraz, null, 2)}

DANE KANAŁU POZA SESJAMI (followersi, suby, tempo wzrostu):
${JSON.stringify(kanal, null, 2)}

NAJNOWSZA SESJA (live: trwa; offline: ostatnia zakończona; "przebieg" to oś czasu: min = minuta od startu, v = średnia widzów, msg = wiadomości w tym odcinku, cz = skumulowana liczba unikalnych czatujących):
${JSON.stringify(currentSession, null, 2)}

HISTORIA SESJI (najnowsze pierwsze; część ma "przebieg"):
${JSON.stringify(sessionsHistory, null, 2)}

WYLICZENIA SERWERA (średnie ważone — nowsze sesje ważą więcej; normalizacja na godzinę; porównania procentowe):
${JSON.stringify(derived, null, 2)}

═══ FORMAT ODPOWIEDZI ═══
Zwróć WYŁĄCZNIE obiekt JSON (bez markdownu i komentarzy) z polami:
- "textResponse": pełny tekst wypowiedzi,
- "audioPrompt": ta sama wypowiedź pod syntezator mowy: czysty tekst bez markdownu, wypunktowań, symboli i emoji, krótkie zdania, liczby i godziny zapisane tak, by dobrze brzmiały na głos po polsku, bez adresów stron. Zachowaj wymaganą długość z sekcji TRYB.
`;
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

// Gemini 3.8 TTS działa przez Interactions API: tekst w input, głos w generation_config.speech_config, format w response_format.
// Styl (live = radiowy, offline = analityczny) próbujemy z krótkim limitem czasu. Jeśli zawiedzie (błąd, brak audio, timeout),
// OD RAZU wracamy do sprawdzonego wariantu bez stylu — i przez 10 minut nie marnujemy czasu na styl. Dzięki temu styl nigdy nie zagłuszy głosu.
let styleOffUntil = 0;
const STYLE_TIMEOUT_MS = Number(process.env.TTS_STYLE_TIMEOUT_MS) || 20000;
async function makeAudio(text, models = TTS_MODELS, mode = 'offline') {
    let lastErr = null;
    const style = TTS_STYLES[mode] || TTS_STYLES.offline;
    for (const model of models) {
        const attempts = Date.now() < styleOffUntil ? [null] : [style, null];
        for (const st of attempts) {
            try {
                const part = { type: 'text', text };
                if (st) part.annotations = [{ type: 'speech_metadata', style: st }];
                const interaction = await withTimeout(ai.interactions.create({
                    model,
                    input: [{ type: 'user_input', content: [part] }],
                    response_format: { type: 'audio' },
                    generation_config: { speech_config: [{ voice: TTS_VOICE }] }
                }), st ? STYLE_TIMEOUT_MS : 30000, `TTS ${model}`);
                const b64 = pickAudio(interaction);
                if (!b64) {
                    lastErr = new Error(`TTS ${model}: brak audio w odpowiedzi`);
                    if (st) { styleOffUntil = Date.now() + 10 * 60e3; console.error(`TTS (${model}) ze stylem bez audio — wyłączam styl na 10 min`); continue; }
                    break;
                }
                const buf = Buffer.from(b64, 'base64');
                // unary zwraca gotowy WAV (RIFF); gdyby przyszedł surowy PCM — dopinamy nagłówek
                const out = buf.slice(0, 4).toString('ascii') === 'RIFF' ? buf : pcmToWav(buf);
                console.log(`TTS OK: ${model}${st ? ` (styl: ${mode})` : ' (bez stylu)'}`);
                return out.toString('base64');
            } catch (e) {
                lastErr = e;
                console.error(`TTS (${model}${st ? ', ze stylem' : ', bez stylu'}) nie zadziałał:`, e.message);
                if (st) { styleOffUntil = Date.now() + 10 * 60e3; continue; } // styl zawiódł → natychmiast wariant bez stylu
                break; // bez stylu też się nie udało → następny model
            }
        }
    }
    if (lastErr) throw lastErr;
    return null;
}

app.get('/', (req, res) => res.send('OK'));
app.get('/health', (req, res) => res.json({ ok: true, textModel: TEXT_MODEL, ttsModels: TTS_MODELS, think: THINK }));

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
        ...TTS_MODELS.map(m => timed('głos ' + m, () => makeAudio('Cześć, test głosu.', [m])))
    ]);
    res.json(out);
});

// Strona testowa głosu: /tts-test  (opcjonalnie ?model=...&text=...&mode=live|offline)
app.get('/tts-test', async (req, res) => {
    const text = String(req.query.text || 'Cześć! Sprawdzamy, czy głos działa poprawnie i brzmi naturalnie po polsku.').slice(0, 600);
    const models = req.query.model ? [String(req.query.model)] : TTS_MODELS;
    const mode = req.query.mode === 'live' ? 'live' : 'offline';
    const t0 = Date.now();
    try {
        const audio = await makeAudio(text, models, mode);
        if (!audio) throw new Error('Brak audio w odpowiedzi');
        res.send(`<meta charset="utf-8"><body style="font:16px sans-serif;background:#111;color:#eee;padding:20px"><p>Model: ${esc(models.join(' → '))} · tryb: ${mode} · czas: ${Date.now() - t0} ms · głos: ${esc(TTS_VOICE)}</p><audio controls autoplay src="data:audio/wav;base64,${audio}"></audio><p>${esc(text)}</p></body>`);
    } catch (e) {
        res.status(500).send(`<meta charset="utf-8"><body style="font:16px sans-serif;background:#111;color:#eee;padding:20px"><p>BŁĄD po ${Date.now() - t0} ms</p><pre style="white-space:pre-wrap;color:#ff8">${esc(e.message)}</pre></body>`);
    }
});

// Osobny endpoint głosu — widżet woła go po otrzymaniu tekstu (mode: live | offline wybiera styl głosu)
app.post('/api/tts', async (req, res) => {
    try {
        const text = String(req.body?.text || '').slice(0, 1500);
        const mode = req.body?.mode === 'live' ? 'live' : 'offline';
        if (!text) return res.status(400).json({ success: false, error: 'Brak tekstu' });
        const audio = await makeAudio(text, TTS_MODELS, mode);
        if (!audio) throw new Error('Model nie zwrócił audio');
        res.json({ success: true, audio });
    } catch (error) {
        console.error('Błąd TTS:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.post('/api/analyze', async (req, res) => {
    try {
        const body = req.body || {};
        const { sessionsHistory = [], currentSession = null, channelName = 'Dolaricious', noData = false } = body;
        const live = body.tryb === 'live' || (body.tryb == null && !!currentSession?.trwaJeszcze);
        const mode = live ? 'live' : 'offline';

        // Zupełny brak danych (ani sesji, ani danych kanału) → krótka gotowa wypowiedź, bez modelu tekstowego
        if (noData) {
            const txt = `Cześć! Tu analityk kanału ${channelName}. Na razie nie mam zapisanych żadnych zakończonych sesji ani danych kanału, więc nie ma jeszcze czego analizować. Po pierwszej zakończonej transmisji omówię wyniki, a potem szerszy obraz polskiego Twitcha.`;
            return res.json({ success: true, analysis: txt, audioPrompt: txt, searchUsed: false, sources: [], mode });
        }

        const cal = calendarContext();
        const derived = derive(currentSession, sessionsHistory);
        const prompt = buildPrompt({ ...body, live, currentSession: withClock(currentSession), sessionsHistory: sessionsHistory.map(withClock) }, cal, derived);
        const lim = LIMITS[mode], think = THINK[mode];

        // UWAGA: wyszukiwanie Google nie działa razem z wymuszonym JSON — JSON parsujemy ręcznie
        let result = null, searchUsed = false, lastErr = null;
        try {
            result = await genRace(prompt, true, lim.search, think);
            searchUsed = true;
        } catch (e) { lastErr = e; console.error('Analiza z wyszukiwaniem nie wyszła:', e.message); }
        if (!result) {
            const fallbackPrompt = prompt + '\n\nUWAGA: wyszukiwarka jest teraz niedostępna. Opieraj się wyłącznie na przesłanych danych i dzisiejszej dacie, a w częściach wymagających sieci (kalendarz i święta, gry, patche, wydarzenia, stosunek widzów do streamujących) powiedz krótko, że bieżących danych z sieci nie udało się sprawdzić. Niczego nie zmyślaj.';
            try { result = await genRace(fallbackPrompt, false, lim.fallback, 'low'); }
            catch (e) { lastErr = e; console.error('Analiza bez wyszukiwania nie wyszła:', e.message); }
        }
        if (!result) throw lastErr || new Error('Analiza nie powiodła się');

        const raw = result.text;
        const parsed = parseJson(raw) || { textResponse: raw, audioPrompt: raw };
        const textResponse = parsed.textResponse || raw;
        const audioPrompt = parsed.audioPrompt || textResponse;

        res.json({ success: true, analysis: textResponse, audioPrompt, searchUsed, sources: result.sources || [], mode });
    } catch (error) {
        console.error('Błąd podczas analizy:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
