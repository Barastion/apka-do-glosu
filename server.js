const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors());

// Klucz pobierany ze zmiennej środowiskowej GEMINI_API_KEY (Render → Environment)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ══════════════ KOLEJNOŚĆ MODELI: ZAWSZE OD NAJLEPSZEGO DO NAJSŁABSZEGO ══════════════
// Modele są próbowane PO KOLEI (następny tylko gdy poprzedni zawiedzie) — nie równolegle, bo to podwajało zużycie limitu.
// Każdy model ma OSOBNE limity, więc po wyczerpaniu jednego serwer przechodzi na następny. Model, którego nie ma (404), jest pomijany na 6 h.
// UWAGA: zmienne TEXT_MODEL, TEXT_MODEL_FALLBACKS i RESEARCH_MODELS ustawione w Render → Environment NADPISUJĄ poniższe listy.
//        Żeby użyć list z kodu, usuń te zmienne w Render.
//
// ANALIZA (najważniejsza; bez wyszukiwania). Limity dzienne z ai.dev/rate-limit:
//   3.8-flash (20) → 3.6-flash (20) → 4-hydrogen (20, potwierdzony w logach) → 3-flash (20) → 3.5-flash-lite (500) → 3.1-flash-lite (500)
//   → Gemma 4 31B / 26B (po 14,4 tys./dobę, ale tylko 16 tys. tokenów/min — ostatnie warstwy).
// Gemini 2.5 (flash i flash-lite) jest na tym koncie WYŁĄCZONY (404 „no longer available to new users”), więc go tu nie ma.
// Gemini Robotics ER 2 celowo NIE jest na liście analizy: jego 20 zapytań dziennie są zarezerwowane dla wyszukiwania w sieci (limit dzienny dotyczy modelu, nie narzędzia).
// Pozycja 4-hydrogen to przypuszczenie co do jakości — jeśli analizy z niego wypadną gorzej, przesuń go niżej (env TEXT_MODEL_FALLBACKS).
// ID modeli 3-flash-preview i gemma-4-* nie były jeszcze potwierdzone: gdy któregoś nie ma (404), serwer pominie go na 6 h.
const TEXT_MODEL = process.env.TEXT_MODEL || 'gemini-3.8-flash';
const list = (v, def) => String(v || def).split(',').map(x => x.trim()).filter(Boolean);
const TEXT_MODELS = [...new Set([TEXT_MODEL, ...list(process.env.TEXT_MODEL_FALLBACKS || process.env.TEXT_MODEL_FALLBACK,
    'gemini-3.6-flash,gemini-4-hydrogen,gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemma-4-31b-it,gemma-4-26b-a4b-it')])];   // sprawdź ID: 3-flash-preview, gemma-4-*
// GŁOS (bez zmian): główny + zapasowe, od najlepszego do najsłabszego; każdy model TTS ma osobny limit dzienny (po 10). Env: TTS_MODEL, TTS_MODEL_FALLBACKS.
const TTS_MODELS = [...new Set([process.env.TTS_MODEL || 'gemini-3.8-flash-tts', ...list(process.env.TTS_MODEL_FALLBACKS, 'gemini-3.8-flash-lite-tts,gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts')])];
const TTS_VOICE = process.env.TTS_VOICE || 'Kore';
// BADANIE SIECI (wyszukiwanie; to tylko dodatek do analizy). Z logów: jedyny potwierdzony model z działającym wyszukiwaniem to Gemini Robotics ER 2 (ok. 12 s).
// Gemini 2.5 jest wyłączony (404), a modele 3.x mają w tabeli limit wyszukiwania 0/0, więc ich tu nie ma.
// Kolejność: robotics-er-2 (sprawdzony) → 4-hydrogen (niepotwierdzony; użyty tylko gdy robotics zawiedzie). Kolejne kandydaty sprawdzisz przez /diag?search=1&models=a,b,c
const RESEARCH_MODELS = [...new Set(list(process.env.RESEARCH_MODELS, 'gemini-robotics-er-2-preview,gemini-4-hydrogen'))];
// modele, które nie przyjmują poziomu „myślenia” (Gemma, seria 2.5) — nie wysyłamy go, żeby nie marnować zapytań na odrzucenia
const noThink = m => /gemma|2\.5/i.test(m);

// Poziom „myślenia” (zmienne środowiskowe THINK_LIVE / THINK_OFFLINE). Liczby są policzone w kodzie, więc 'low' w zupełności wystarcza na żywo.
const THINK = { live: process.env.THINK_LIVE || 'low', offline: process.env.THINK_OFFLINE || 'low' };

// Czas: KAŻDY model ma własne 45 s (PER_MODEL_MS). Gdy model zwróci błąd (np. 429), przechodzimy do następnego od razu, bez czekania.
// Natychmiastowe odrzucenia (429 / 404) NIE zużywają limitu prób — liczą się tylko timeouty i błędy serwera (MAX_ATTEMPTS).
const PER_MODEL_MS = Number(process.env.PER_MODEL_MS) || 45000;
const MAX_ATTEMPTS = Number(process.env.MAX_ATTEMPTS) || 3;
// Budżety czasu (aplikacja czeka na analizę maks. 285 s, więc całość mieści się z zapasem)
const RESEARCH_MS = Number(process.env.RESEARCH_MS) || 60000;                 // jeden model w badaniu sieci (wyszukiwanie trwa dłużej)
const RESEARCH_BUDGET_MS = Number(process.env.RESEARCH_BUDGET_MS) || 100000;   // całe badanie sieci
const TOTAL_BUDGET_MS = Number(process.env.TOTAL_BUDGET_MS) || 240000;         // badanie + analiza razem
// TWARDY LIMIT blokady modelu: nawet gdy Gemini napisze „spróbuj za 3 dni”, próbujemy ponownie najpóźniej po 12 h
const MAX_BLOCK_MS = Number(process.env.MAX_BLOCK_MS) || 12 * 3600e3;
// Blokada krótsza niż to jest „miękka” (limit minutowy / chwilowy błąd) — gdy wszystkie modele są wstrzymane, próbujemy je mimo blokady
const SOFT_BLOCK_MS = 15 * 60e3;

// Czas życia pamięci podręcznej
const TTL = {
    analysisLive: 15 * 60e3,      // na żywo: ta sama analiza (i nagranie) przez 15 min dla tej samej gry i tytułu (0 zapytań)
    analysisOffline: 60 * 60e3,   // offline: ta sama analiza przez 1 h
    games: 12 * 3600e3,           // notatki z sieci (gry, Twitch, wydarzenia w kraju) ważne 12 h w ciągu tej samej doby
    audio: 2 * 3600e3             // nagrania kasowane po 2 h
};

// Style głosu: na żywo — radiowy, lekko komentatorski; offline — analityczny (głos nadal kobiecy: Kore)
// Do obu stylów dopisane zdanie o stałym głosie; CTA na końcu live jest mówione dokładnie tak samo jak reszta.
const SAME_VOICE = 'Keep exactly the same voice, pitch, timbre and pace from the first word to the last.';
const TTS_STYLES = {
    live: `warm, energetic female radio presenter, lightly sports-commentator style, engaging and upbeat, lively pace; the closing call-to-action sentences are delivered in exactly the same voice and manner as the rest of the speech. ${SAME_VOICE}`,
    offline: `calm, analytical, professional female voice, clear and measured pace, natural and friendly. ${SAME_VOICE}`
};

const isQuota = e => /429|RESOURCE_EXHAUSTED|exceeded your current quota/i.test(String(e?.message || e));
// Model, który zwrócił 429 (limit) lub 404 (nie istnieje), jest chwilowo pomijany — bez marnowania czasu i zapytań na odrzucane próby
const blocked = new Map();
// Blokowanie dla GŁOSU (bez zmian)
function blockFor(e) {
    const m = String(e?.message || e);
    if (/not found|NOT_FOUND|no longer available|\b404\b/i.test(m)) return 6 * 3600e3;          // model nie istnieje → 6 h
    if (isQuota(e)) return /PerDay|per day|daily/i.test(m) ? 30 * 60e3 : 90e3;                    // limit dzienny → 30 min, minutowy → 90 s
    return 0;
}
const isBlocked = m => (blocked.get(m) || 0) > Date.now();
setInterval(() => { const n = Date.now(); for (const [k, t] of blocked) if (t <= n) blocked.delete(k); }, 10 * 60e3).unref();

// Blokowanie dla modeli TEKSTOWYCH: czas z komunikatu Google („retry in Xs”), a przy powtarzających się 429 coraz dłużej (1 min → 5 → 30 → 60)
const strikes = new Map(); // model -> { n, t }
const retryHint = e => {
    const m = String(e?.message || e);
    const a = m.match(/retry in ((?:[\d.]+\s*h)?\s*(?:[\d.]+\s*m(?!s))?\s*(?:[\d.]+\s*s)?)/i) || m.match(/"retryDelay"\s*:\s*"([^"]+)"/i);
    if (!a) return 0;
    const t = a[1], g = re => { const x = t.match(re); return x ? Number(x[1]) : 0; };
    const ms = (g(/([\d.]+)\s*h/i) * 3600 + g(/([\d.]+)\s*m(?!s)/i) * 60 + g(/([\d.]+)\s*s/i)) * 1000;
    return ms ? Math.min(Math.ceil(ms) + 2000, MAX_BLOCK_MS) : 0;
};
function blockForText(e, model) {
    const m = String(e?.message || e), now = Date.now();
    if (/not found|NOT_FOUND|no longer available|\b404\b/i.test(m)) return 6 * 3600e3;
    // wyszukiwanie: model, który nie przyjmuje narzędzia Google Search (400), pomijamy na 6 h (blokada dotyczy tylko wyszukiwania, klucz „search|model”)
    if (String(model).startsWith('search|') && /\b400\b|INVALID_ARGUMENT|not supported|unsupported/i.test(m)) return 6 * 3600e3;
    if (isQuota(e)) {
        if (/limit:\s*0\b/i.test(m)) return 6 * 3600e3;   // „limit: 0” = model nie ma w ogóle limitu na tym koncie → nie ma sensu próbować co minutę
        const s = strikes.get(model), n = (s && now - s.t < 2 * 3600e3) ? s.n + 1 : 1;
        strikes.set(model, { n, t: now });
        if (/PerDay|per day|daily/i.test(m)) return retryHint(e) || 60 * 60e3;   // limit dzienny: czas od Google, ale max MAX_BLOCK_MS
        return Math.max(retryHint(e), [60e3, 3 * 60e3, 10 * 60e3][Math.min(n - 1, 2)]);
    }
    if (/przekroczono limit/.test(m)) return 2 * 60e3;                                            // timeout → model odpoczywa 2 min
    if (/\b(500|503)\b|UNAVAILABLE|overloaded/i.test(m)) return 2 * 60e3;
    return 0;
}
setInterval(() => { const n = Date.now(); for (const [k, v] of strikes) if (n - v.t > 2 * 3600e3) strikes.delete(k); }, 30 * 60e3).unref();

const blockedSummary = () => [...blocked].filter(([, t]) => t > Date.now()).map(([k, t]) => `${k} (${Math.ceil((t - Date.now()) / 60e3)} min)`).join(', ');
const withTimeout = (p, ms, label) => Promise.race([
    p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label}: przekroczono limit ${Math.round(ms / 1000)}s`)), ms))
]);

// ═══════════════ PAMIĘĆ PODRĘCZNA (przechowuje też obietnice w toku, więc równoległe kliknięcia nie dublują zapytań) ═══════════════
function makeCache(max) {
    const m = new Map();
    // co 10 minut usuwamy przeterminowane wpisy, nawet jeśli nikt już o nie nie pyta
    setInterval(() => { const now = Date.now(); for (const [k, e] of m) if (e.exp <= now) m.delete(k); }, 10 * 60e3).unref();
    return {
        del: k => m.delete(k),
        async memo(k, ttl, fn) {
            const e = m.get(k);
            if (e && e.exp > Date.now()) return e.val;
            m.delete(k);
            const p = fn();
            m.set(k, { exp: Date.now() + ttl, val: p });
            while (m.size > max) m.delete(m.keys().next().value);
            try { return await p; } catch (err) { m.delete(k); throw err; }
        }
    };
}
const cacheMain = makeCache(60);
const cacheAudio = makeCache(10);

// Zwarty JSON dla promptu: bez wcięć oraz bez null / pustych napisów / pustych tablic (mniej tokenów wejściowych)
const J = o => JSON.stringify(o, (k, v) => (v === null || v === '' || (Array.isArray(v) && !v.length)) ? undefined : v);

// ═══════════════ WYWOŁANIA MODELU ═══════════════
// Klasyczne generateContent (plan B dla Interactions API)
async function gen(prompt, search, ms, model, think) {
    const base = search ? { tools: [{ googleSearch: {} }] } : {};
    const tries = think ? [{ ...base, thinkingConfig: { thinkingLevel: think } }, base] : [base];
    let last = null;
    for (const config of tries) {
        try {
            return await withTimeout(ai.models.generateContent({ model, contents: prompt, config }), ms, `Analiza (${model})`);
        } catch (e) {
            last = e;
            if (/przekroczono limit/.test(e.message) || isQuota(e)) throw e;
            console.error('Próba generowania nieudana:', e.message);
        }
    }
    throw last;
}

function collectSources(i) {
    const out = new Set();
    for (const st of i?.steps || []) for (const c of st.content || []) for (const a of c.annotations || []) if (a?.url) out.add(a.url);
    return [...out];
}

async function genOne(prompt, search, ms, model, think) {
    try {
        const th = noThink(model) ? null : think;
        const req = { model, input: prompt };
        if (th) req.generation_config = { thinking_level: th };
        if (search) req.tools = [{ type: 'google_search' }];
        const i = await withTimeout(ai.interactions.create(req, { maxRetries: 0 }), ms, `Analiza (${model})`);   // maxRetries:0 — SDK domyślnie ponawia 429 do 4 razy (do 30 s), a to marnuje czas i zapytania
        if (!i.output_text) throw new Error('pusta odpowiedź');
        return { text: i.output_text, model, via: 'interactions', sources: collectSources(i) };
    } catch (e) {
        if (/przekroczono limit/.test(e.message) || isQuota(e)) throw e;
        // drugie zapytanie (generateContent) tylko gdy Interactions go nie obsługuje; przy 404 i 5xx to strata zapytania
        if (!/\b400\b|INVALID_ARGUMENT|not supported|pusta odpowiedź|TypeError|Cannot read|is not a function/i.test(e.message)) throw e;
        console.error(`Interactions (${model}) nie wyszło:`, e.message);
    }
    const r = await gen(prompt, search, ms, model, noThink(model) ? null : think);
    if (!String(r.text || '').trim()) throw new Error('pusta odpowiedź');   // pusty wynik = porażka tego modelu, serwer idzie do następnego
    const chunks = r.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    return { text: r.text, model, via: 'generateContent', sources: chunks.map(c => c.web?.uri).filter(Boolean) };
}

// Modele PO KOLEI: każdy dostaje pełne `ms` (nie więcej niż zostało z budżetu); błąd = od razu następny; pomijamy modele wstrzymane.
// Odrzucenia 429 i 404 przychodzą od razu, więc nie zużywają limitu prób.
// Blokady: limit samego modelu blokuje model wszędzie; ogólny 429 przy WYSZUKIWANIU blokuje model tylko dla wyszukiwania (klucz 'search|model'),
// bo wyczerpany limit wyszukiwarki nie oznacza, że model nie napisze analizy bez niej.
// Gdy WSZYSTKIE modele są wstrzymane krótką („miękką”) blokadą, robimy drugie przejście mimo blokad, zamiast od razu zwracać błąd.
async function genSeq(prompt, search, ms, think, models = TEXT_MODELS, maxAttempts = MAX_ATTEMPTS, deadline = Infinity, accept = null) {
    const t0 = Date.now(), errs = [], triedSet = new Set();
    let attempts = 0, tried = 0;
    const remaining = m => Math.max(blocked.get(m) || 0, search ? (blocked.get('search|' + m) || 0) : 0) - Date.now();
    const pass = async force => {
        for (const m of models) {
            if (attempts >= maxAttempts) return null;
            const left = deadline - Date.now();
            if (left < 8000) { errs.push('budżet czasu wyczerpany'); return null; }
            if (force === 'all' && triedSet.has(m)) continue;   // ostatnia szansa: tylko modele, których jeszcze nie próbowaliśmy
            const rem = remaining(m);
            if (rem > 0 && force !== 'all' && (!force || rem > SOFT_BLOCK_MS)) {
                if (!force) errs.push(`${m}: 429 RESOURCE_EXHAUSTED — model wstrzymany po przekroczeniu limitu (jeszcze ok. ${Math.ceil(rem / 60e3)} min)`);
                continue;
            }
            attempts++; tried++; triedSet.add(m);
            try {
                const r = await genOne(prompt, search, Math.min(ms, left), m, think);
                if (accept && !accept(r)) throw new Error('pusta lub nieczytelna odpowiedź');   // zła odpowiedź = następny model, a nie awaria całej analizy
                blocked.delete(m); blocked.delete('search|' + m); strikes.delete(m); strikes.delete('search|' + m);
                console.log(`Model OK (${r.model}, ${r.via}${search ? ', z wyszukiwaniem' : ''}, think=${think}): ${Date.now() - t0} ms`);
                return r;
            } catch (e) {
                const msg = String(e?.message || e);
                errs.push(`${m}: ${msg}`); console.error('Model zawiódł:', msg);
                const key = search ? 'search|' + m : m;   // porażka WYSZUKIWANIA blokuje model tylko dla wyszukiwania — nigdy dla analizy
                const b = Math.min(blockForText(e, key), MAX_BLOCK_MS);
                if (b) blocked.set(key, Date.now() + b);
                if (isQuota(e) || /not found|NOT_FOUND|\b404\b/i.test(msg)) attempts--;   // natychmiastowe odrzucenie nie zużywa próby
            }
        }
        return null;
    };
    let r = await pass(false);
    if (!r && tried === 0 && deadline - Date.now() >= 8000) { console.log('Wszystkie modele wstrzymane krótką blokadą — próbuję mimo to'); r = await pass(true); }
    // Analiza jest najważniejsza: gdy wszystko wstrzymane lub zawiodło, na końcu próbujemy jeszcze modele mimo blokad (odrzucenie 429 przychodzi od razu, więc to nic nie kosztuje)
    if (!r && !search && deadline - Date.now() >= 8000 && triedSet.size < models.length) { console.log('Analiza: ostatnia szansa — próbuję pozostałe modele mimo blokad'); r = await pass('all'); }
    if (r) return r;
    throw new Error(errs.join(' | '));
}

function parseJson(t) {
    const c = String(t || '').replace(/```json|```/gi, '').trim();
    try { return JSON.parse(c); } catch (e) {}
    const m = c.match(/\{[\s\S]*\}/);
    if (m) { try { return JSON.parse(m[0]); } catch (e) {} }
    return null;
}

// Czysty tekst pod syntezator mowy: bez markdownu, przypisów, adresów
const cleanSpeech = t => String(t || '')
    .replace(/```[a-z]*/gi, '').replace(/\[\d+(?:\s*,\s*\d+)*\]/g, '').replace(/https?:\/\/\S+/g, '')
    .replace(/[*#`>]+/g, '').replace(/^\s*[-•]\s+/gm, '').replace(/\s*\n+\s*/g, ' ').replace(/\s{2,}/g, ' ').trim();
// Tekst do wypowiedzenia z odpowiedzi modelu (JSON z audioPrompt/textResponse albo zwykły tekst). Zwraca '' gdy odpowiedź jest pusta lub zepsuta.
const speechFrom = raw => {
    try {
        const t = String(raw || '').trim();
        if (!t) return '';
        const parsed = t.startsWith('{') ? parseJson(t) : null;
        if (t.startsWith('{') && !parsed) return '';                       // niedokończony / zepsuty JSON — nie czytamy go na głos
        const out = cleanSpeech(parsed?.audioPrompt || parsed?.textResponse || t);
        return out.length >= 20 ? out : '';
    } catch (e) { return ''; }
};

// Usuwa z danych pola, które wyglądają na nicki/listy użytkowników (po nazwie klucza; liczby i wartości logiczne zostają).
// Model w ogóle ich nie dostaje, więc nie może ich przeczytać. Wzorzec można rozszerzyć, gdy widżet doda nowe pola.
const NICK_KEY = /nick|login|username|użytkownik|uzytkownik|chatter|raider|rajder|hostuj|top(czat|chat|follow|sub|donat|cheer)|najaktywn|donator|gifter|cheerer/i;
function stripNicks(v) {
    if (Array.isArray(v)) return v.map(stripNicks);
    if (v && typeof v === 'object') {
        const o = {};
        for (const [k, x] of Object.entries(v)) {
            if (NICK_KEY.test(k) && typeof x !== 'number' && typeof x !== 'boolean') continue;
            o[k] = stripNicks(x);
        }
        return o;
    }
    return v;
}

// Skoki widowni mniejsze niż SPIKE_MIN osób (także zapisane wcześniej w historii aplikacji) nie liczą się jako rajd/hosting
const keepBigSpikes = s => (s && Array.isArray(s.skokiWidzow))
    ? { ...s, skokiWidzow: s.skokiWidzow.filter(k => k && (k.szczyt - (k.przedSkokiem || 0)) >= SPIKE_MIN) } : s;

// ═══════════════ DATA I CZAS ═══════════════
const TZ = 'Europe/Warsaw';
function warsawParts(d) {
    const f = new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'long', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
    return Object.fromEntries(f.formatToParts(d).map(x => [x.type, x.value]));
}
const warsawYmd = d => { const p = warsawParts(d); return `${p.year}-${p.month}-${p.day}`; };
const dayDiff = (a, b) => Math.round((Date.parse(warsawYmd(a) + 'T00:00:00Z') - Date.parse(warsawYmd(b) + 'T00:00:00Z')) / 864e5);
const clock = d => new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(d);

// Polska odmiana: 1 → one, 2–4 (poza 12–14) → few, reszta → many
const plural = (n, one, few, many) => {
    const a = Math.abs(n), l = a % 10, t = a % 100;
    if (a === 1) return one;
    return (l >= 2 && l <= 4 && !(t >= 12 && t <= 14)) ? few : many;
};
// Gotowa fraza „ile temu” dla modelu (żeby nie przeliczał sam i nie zaokrąglał do „0 godzin”):
// poniżej godziny — dokładne minuty; od godziny — godziny + kwadranse; od doby — „wczoraj” / „N dni temu”
function temu(thenMs, now = new Date()) {
    const min = Math.max(0, Math.round((now.getTime() - thenMs) / 6e4));
    if (min < 1) return 'przed chwilą';
    if (min < 60) return `${min} ${plural(min, 'minutę', 'minuty', 'minut')} temu`;
    const total = Math.round(min / 15) * 15;
    if (total >= 1440) {
        const dd = Math.max(1, dayDiff(now, new Date(thenMs)));
        return dd === 1 ? 'wczoraj' : `${dd} dni temu`;
    }
    const h = Math.floor(total / 60), m = total % 60;
    const hTxt = `${h === 1 ? '' : h + ' '}${plural(h, 'godzinę', 'godziny', 'godzin')}`;
    return `${hTxt}${m ? ` i ${m} minut` : ''} temu`;
}

// Gotowa fraza „jak długo trwał stream” (np. „4 godziny i 30 minut”) — model NIE ma sam dodawać ani przeliczać czasów.
// Zaokrąglenie do 5 minut, żeby fraza brzmiała naturalnie.
function durFraza(totalMin) {
    const t = Math.max(0, Math.round((totalMin || 0) / 5) * 5);
    if (t < 1) return 'mniej niż minutę';
    const h = Math.floor(t / 60), m = t % 60;
    const hTxt = h ? `${h} ${plural(h, 'godzina', 'godziny', 'godzin')}` : '';
    const mTxt = m ? `${m} ${plural(m, 'minuta', 'minuty', 'minut')}` : '';
    return [hTxt, mTxt].filter(Boolean).join(' i ');
}
const sessMin = s => s && (s.czasTrwaniaMin > 0 ? s.czasTrwaniaMin : (s.start && s.koniec ? Math.max(0, (Date.parse(s.koniec) - Date.parse(s.start)) / 6e4) : 0));

// Godzina 24-godzinna zapisana słownie w formie „od … / do … / o …” (np. 17:30 → „siedemnastej trzydzieści”).
// Model dostaje gotowe frazy i ma je przepisywać — nie zamienia sam „17:30” na „piąta trzydzieści”.
const GODZ_DOP = ['północy', 'pierwszej', 'drugiej', 'trzeciej', 'czwartej', 'piątej', 'szóstej', 'siódmej', 'ósmej', 'dziewiątej', 'dziesiątej', 'jedenastej', 'dwunastej', 'trzynastej', 'czternastej', 'piętnastej', 'szesnastej', 'siedemnastej', 'osiemnastej', 'dziewiętnastej', 'dwudziestej', 'dwudziestej pierwszej', 'dwudziestej drugiej', 'dwudziestej trzeciej'];
// Liczby 0–59 słownie (minuty): 5 → „pięć”, 43 → „czterdzieści trzy”; minuty 1–9 po godzinie mówimy z „zero” („zero pięć”)
const UN_SLOW = ['zero', 'jeden', 'dwa', 'trzy', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć'];
const TEEN_SLOW = ['dziesięć', 'jedenaście', 'dwanaście', 'trzynaście', 'czternaście', 'piętnaście', 'szesnaście', 'siedemnaście', 'osiemnaście', 'dziewiętnaście'];
const TENS_SLOW = ['', '', 'dwadzieścia', 'trzydzieści', 'czterdzieści', 'pięćdziesiąt'];
const liczSlownie = n => n < 10 ? UN_SLOW[n] : n < 20 ? TEEN_SLOW[n - 10] : TENS_SLOW[Math.floor(n / 10)] + (n % 10 ? ' ' + UN_SLOW[n % 10] : '');
const minSlownie = m => m < 10 ? 'zero ' + UN_SLOW[m] : liczSlownie(m);
function godzSlownie(hhmm) {
    const [h, m] = String(hhmm).split(':').map(Number);
    if (!Number.isFinite(h) || !Number.isFinite(m)) return String(hhmm);
    if (h % 24 === 0 && m === 30) return 'pół godziny po północy';
    const base = GODZ_DOP[h % 24];
    return m ? `${base} ${minSlownie(m)}` : base;
}
// Godzina w mianowniku („godzina dwudziesta pierwsza dwadzieścia trzy”) i data słownie („wtorek, siedemnasty listopada”) — gotowe frazy na początek audycji
const GODZ_MIA = ['zero', 'pierwsza', 'druga', 'trzecia', 'czwarta', 'piąta', 'szósta', 'siódma', 'ósma', 'dziewiąta', 'dziesiąta', 'jedenasta', 'dwunasta', 'trzynasta', 'czternasta', 'piętnasta', 'szesnasta', 'siedemnasta', 'osiemnasta', 'dziewiętnasta', 'dwudziesta', 'dwudziesta pierwsza', 'dwudziesta druga', 'dwudziesta trzecia'];
const godzMianownik = (h, m) => (h === 0 && m === 0) ? 'północ' : GODZ_MIA[h % 24] + (m ? ' ' + minSlownie(m) : '');
const DZIEN_PORZ = ['', 'pierwszy', 'drugi', 'trzeci', 'czwarty', 'piąty', 'szósty', 'siódmy', 'ósmy', 'dziewiąty', 'dziesiąty', 'jedenasty', 'dwunasty', 'trzynasty', 'czternasty', 'piętnasty', 'szesnasty', 'siedemnasty', 'osiemnasty', 'dziewiętnasty', 'dwudziesty', 'dwudziesty pierwszy', 'dwudziesty drugi', 'dwudziesty trzeci', 'dwudziesty czwarty', 'dwudziesty piąty', 'dwudziesty szósty', 'dwudziesty siódmy', 'dwudziesty ósmy', 'dwudziesty dziewiąty', 'trzydziesty', 'trzydziesty pierwszy'];
const okresSlownie = (a, b) => `od ${godzSlownie(a)} do ${godzSlownie(b)}`;

// Jedno źródło prawdy o czasie sesji: start i koniec (ISO). Pola z widżetu (czasTrwaniaMin, godzinaStartu, godzinaKonca, dzienTygodnia)
// bywają niespójne, więc je sprawdzamy i w razie rozbieżności przeliczamy. Wszystkie rozbieżności trafiają do ostrzeżeń (log + odpowiedź).
// Wyłączenie poprawiania czasu trwania: zmienna środowiskowa CZAS_Z_KONCA=0.
const DUR_FROM_SPAN = process.env.CZAS_Z_KONCA !== '0';
const hhmmNorm = v => String(v || '').trim().replace(/^(\d):/, '0$1:').slice(0, 5);
function fixSession(s, warns = []) {
    if (!s || !s.start) return s;
    const t0 = Date.parse(s.start);
    if (!Number.isFinite(t0)) { warns.push(`sesja z niepoprawnym startem: ${s.start}`); return s; }
    const o = { ...s }, tag = `${warsawYmd(new Date(t0))} ${clock(new Date(t0))}`;
    const dur = Number(s.czasTrwaniaMin) || 0;
    const e = s.koniec ? Date.parse(s.koniec) : (s.trwaJeszcze ? Date.now() : NaN);
    let t1 = null;
    if (Number.isFinite(e)) {
        const span = (e - t0) / 6e4;
        if (span > 0 && span <= 24 * 60) {
            t1 = e;
            if (span > 14 * 60) warns.push(`sesja ${tag}: podejrzanie długa (${Math.round(span / 60)} h) — sprawdź zapis startu/końca w widżecie`);
            if (!(dur > 0) || Math.abs(span - dur) > Math.max(10, dur * 0.1)) {
                warns.push(`sesja ${tag}: czasTrwaniaMin=${dur} z widżetu, a start→koniec = ${Math.round(span)} min${DUR_FROM_SPAN ? ' — użyto start→koniec' : ' (poprawka wyłączona)'}`);
                if (DUR_FROM_SPAN) o.czasTrwaniaMin = Math.round(span);
            }
        } else if (s.koniec) warns.push(`sesja ${tag}: koniec (${s.koniec}) nielogiczny względem startu — pominięto`);
    }
    const st = clock(new Date(t0));
    if (s.godzinaStartu && hhmmNorm(s.godzinaStartu) !== st) warns.push(`sesja ${tag}: godzinaStartu z widżetu=${s.godzinaStartu}, z pola start=${st} — użyto ${st}`);
    o.godzinaStartu = st;
    const endMs = t1 ?? t0 + (o.czasTrwaniaMin || 0) * 6e4;
    if (s.koniec || s.godzinaKonca || o.czasTrwaniaMin > 0) {
        const en = clock(new Date(endMs));
        if (s.godzinaKonca && hhmmNorm(s.godzinaKonca) !== en) warns.push(`sesja ${tag}: godzinaKonca z widżetu=${s.godzinaKonca}, wyliczona=${en} — użyto ${en}`);
        if (!s.trwaJeszcze) o.godzinaKonca = en;
    }
    const wd = warsawParts(new Date(t0)).weekday;
    if (s.dzienTygodnia && String(s.dzienTygodnia).toLowerCase() !== wd) warns.push(`sesja ${tag}: dzienTygodnia z widżetu=${s.dzienTygodnia}, z pola start=${wd} — użyto ${wd}`);
    o.dzienTygodnia = wd;
    return o;
}

// ═══════════════ KALENDARZ (liczony w kodzie z reguł — działa w każdym roku bez zmian w pliku) ═══════════════
// Święta stałe mają stałe daty, ruchome (Wielkanoc, Poniedziałek Wielkanocny, Zielone Świątki, Boże Ciało) wynikają z jednego wzoru na Wielkanoc.
// Wigilia jest dniem wolnym od 2025 roku. Zdarzenia nadzwyczajne (żałoba narodowa, dodatkowy dzień wolny, wybory) sprawdza wąskie zapytanie do sieci.
function easterUtc(y) {   // algorytm gregoriański (Meeusa/Jonesa/Butchera) → północ UTC niedzieli wielkanocnej
    const a = y % 19, b = Math.floor(y / 100), c = y % 100, d = Math.floor(b / 4), e = b % 4, f = Math.floor((b + 8) / 25), g = Math.floor((b - f + 1) / 3),
        h = (19 * a + b - d - g + 15) % 30, i = Math.floor(c / 4), k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = Math.floor((a + 11 * h + 22 * l) / 451),
        x = h + l - 7 * m + 114;
    return Date.UTC(y, Math.floor(x / 31) - 1, (x % 31) + 1);
}
const ymdU = t => new Date(t).toISOString().slice(0, 10);
const holidayCache = {};
function holidays(y) {
    if (holidayCache[y]) return holidayCache[y];
    const E = easterUtc(y), D = 864e5, h = {};
    const add = (t, n) => { h[ymdU(t)] = n; }, fx = (m, d, n) => add(Date.UTC(y, m - 1, d), n);
    fx(1, 1, 'Nowy Rok'); fx(1, 6, 'Trzech Króli'); add(E, 'Wielkanoc'); add(E + D, 'Poniedziałek Wielkanocny');
    fx(5, 1, 'Święto Pracy'); fx(5, 3, 'Święto Konstytucji 3 Maja'); add(E + 49 * D, 'Zielone Świątki'); add(E + 60 * D, 'Boże Ciało');
    fx(8, 15, 'Wniebowzięcie Najświętszej Maryi Panny'); fx(11, 1, 'Wszystkich Świętych'); fx(11, 11, 'Narodowe Święto Niepodległości');
    if (y >= 2025) fx(12, 24, 'Wigilia Bożego Narodzenia');
    fx(12, 25, 'Boże Narodzenie (pierwszy dzień)'); fx(12, 26, 'Boże Narodzenie (drugi dzień)');
    return holidayCache[y] = h;
}
const holidayName = t => holidays(new Date(t).getUTCFullYear())[ymdU(t)] || null;
const isFreeDay = t => { const w = new Date(t).getUTCDay(); return w === 0 || w === 6 || !!holidayName(t); };
const isWeekend = t => { const w = new Date(t).getUTCDay(); return w === 0 || w === 6; };
const dayPl = t => DAY_PL[(new Date(t).getUTCDay() + 6) % 7];
const DAY_GEN = ['poniedziałku', 'wtorku', 'środy', 'czwartku', 'piątku', 'soboty', 'niedzieli'];
const dayGen = t => DAY_GEN[(new Date(t).getUTCDay() + 6) % 7];
const ZWYKLY = ['zwykły poniedziałek', 'zwykły wtorek', 'zwykłą środę', 'zwykły czwartek', 'zwykły piątek', 'zwykłą sobotę', 'zwykłą niedzielę'];
const dateFmtU = new Intl.DateTimeFormat('pl-PL', { timeZone: 'UTC', day: 'numeric', month: 'long' });
const dateU = t => dateFmtU.format(new Date(t));
const OKAZJE = { '02-14': 'Walentynki', '03-08': 'Dzień Kobiet', '06-01': 'Dzień Dziecka', '10-31': 'Halloween', '11-02': 'Zaduszki', '12-06': 'Mikołajki', '12-31': 'Sylwester' };
// Ogólny nastrój miesiąca w Polsce (zamiast liczenia długości dnia): pora roku, światło, pogoda i rytm szkoły/pracy — czyli rzeczy, które trwają CAŁY miesiąc.
// Pojedyncze wydarzenia (majówka, matury, Walentynki, Halloween, mecze, święta) NIE wchodzą do opisów miesięcy: święta i dni wolne liczy kalendarz, resztę wyłapuje zapytanie do sieci.
// Wyjątki to okresy rozciągnięte w czasie, które zmieniają się w trakcie miesiąca (koniec roku szkolnego i wakacje, koniec wakacji, grudzień).
const MIESIACE = ['styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec', 'sierpień', 'wrzesień', 'październik', 'listopad', 'grudzień'];
const MIESIAC_NOTKA = [
    'środek zimy, mroźno i bardzo krótkie dni, ludzie dużo czasu spędzają w domu',
    'końcówka zimy, zimno i krótkie dni (choć stopniowo coraz dłuższe), ludzie dużo czasu spędzają w domu',
    'przełom zimy i wiosny, dni wyraźnie dłuższe, pierwsze cieplejsze dni, ludzie stopniowo spędzają coraz więcej czasu na zewnątrz',
    'wiosna, dni długie i coraz cieplejsze, coraz więcej wyjść na zewnątrz i weekendowych wyjazdów',
    'późna wiosna, ciepło i długie dni, wielu ludzi spędza czas na zewnątrz',
    null,   // czerwiec: zależnie od dnia (monthNote)
    'wakacje, upały i bardzo długie jasne wieczory, ludzie podróżują i mają dużo wolnego czasu (więcej widzów, ale i więcej streamujących), wielu jest poza domem',
    null,   // sierpień: zależnie od dnia (monthNote)
    'powrót do szkoły i pracy po wakacjach, dni coraz krótsze, ludzie wracają do domowej rutyny',
    'jesień, robi się zimno, dni wyraźnie krótsze i coraz szybciej ciemno, ludzie coraz więcej czasu spędzają w domu',
    'późna jesień, zimno i szaro, bardzo wcześnie robi się ciemno, ludzie dużo czasu spędzają w domu',
    null    // grudzień: zależnie od dnia (monthNote)
];
function monthNote(month, day) {
    if (month === 6) return day <= 20
        ? 'początek lata, bardzo długie jasne wieczory, ludzie coraz częściej poza domem; rok szkolny jeszcze trwa'
        : 'koniec roku szkolnego i początek wakacji, bardzo długie jasne wieczory, ludzie coraz częściej poza domem';
    if (month === 8) return day <= 24
        ? 'wakacje i urlopy, wyjazdy, jasne wieczory, ludzie mają dużo wolnego czasu'
        : 'koniec wakacji, powroty z wyjazdów, wieczory robią się krótsze, zbliża się rok szkolny';
    if (month === 12) return day <= 23
        ? 'zima, najmniej słońca w roku i bardzo wcześnie ciemno; trwają przygotowania świąteczne, ludzie dużo czasu spędzają w domu'
        : day <= 26 ? 'święta Bożego Narodzenia, rodzinne spotkania, wielu ludzi ma wolne'
        : 'przerwa świąteczna między świętami a Nowym Rokiem, wielu ludzi ma wolne';
    return MIESIAC_NOTKA[month - 1];
}

function calendarBlock(now = new Date()) {
    const p = warsawParts(now), t0 = Date.UTC(+p.year, +p.month - 1, +p.day), D = 864e5;
    const hn = holidayName(t0), wk = isWeekend(t0), idx = (new Date(t0).getUTCDay() + 6) % 7, dn = dayPl(t0);
    const lines = [];
    // 1) typ dnia
    if (hn && !wk) lines.push(`Dziś: ${dn}, dzień wolny od pracy (święto: ${hn}). Widownia i liczba streamerów zwykle wyższe niż w ${ZWYKLY[idx]}.`);
    else if (hn && wk) lines.push(`Dziś: ${dn} — weekend, a zarazem święto (${hn}). Widownia i liczba streamerów zwykle wyższe niż w dni robocze.`);
    else if (wk) lines.push(`Dziś: ${dn} — weekend. Widownia i liczba streamerów zwykle wyższe niż w dni robocze.`);
    else lines.push(`Dziś: ${dn} — zwykły dzień roboczy.`);
    const occ = OKAZJE[`${p.month}-${p.day}`]; if (occ) lines.push(`Okazja dnia: ${occ}.`);
    // 2) długi weekend / most (ciągły blok dni wolnych zawierający święto)
    const block = t => { let a = t, b = t; while (isFreeDay(a - D)) a -= D; while (isFreeDay(b + D)) b += D; return { a, b, len: (b - a) / D + 1 }; };
    const hasHol = (a, b) => { for (let t = a; t <= b; t += D) if (holidayName(t)) return true; return false; };
    const isBridge = t => !isFreeDay(t) && isFreeDay(t - D) && isFreeDay(t + D) && (holidayName(t - D) || holidayName(t + D));
    // ciąg dni wolnych razem z „mostami” (dzień roboczy między dniem wolnym a weekendem): pokazuje, ile wolnego wychodzi, gdy ktoś weźmie dzień urlopu
    const withBridges = t => { const fr = x => isFreeDay(x) || isBridge(x); let a = t, b = t; while (fr(a - D)) a -= D; while (fr(b + D)) b += D; return { a, b, len: (b - a) / D + 1 }; };
    if (isFreeDay(t0)) {
        const bl = block(t0);
        if (bl.len >= 3 && hasHol(bl.a, bl.b)) lines.push(`Długi weekend: ${bl.len} dni wolnych z rzędu (od ${dayGen(bl.a)}, ${dateU(bl.a)}, do ${dayGen(bl.b)}, ${dateU(bl.b)}).`);
        const ex = withBridges(t0);
        if (ex.len > bl.len) {
            const br = []; for (let t = ex.a; t <= ex.b; t += D) if (isBridge(t)) br.push(`${dayPl(t)}, ${dateU(t)}`);
            lines.push(`Z „mostem” (${br.join('; ')}) wolne może trwać łącznie ${ex.len} dni (od ${dayGen(ex.a)}, ${dateU(ex.a)}, do ${dayGen(ex.b)}, ${dateU(ex.b)}) — wielu ludzi bierze wtedy urlop.`);
        }
    } else {
        if (isBridge(t0)) lines.push('Dziś jest „most” — dzień roboczy między dniem wolnym a weekendem; wiele osób bierze wolne, więc ruch bywa jak w dzień wolny.');
        else if (isFreeDay(t0 + D)) { const bl = block(t0 + D); if (bl.len >= 3 && hasHol(bl.a, bl.b)) lines.push(`Jutro zaczyna się długi weekend (${bl.len} dni wolnych z rzędu) — dziś wielu ludzi kończy tydzień wcześniej.`); }
    }
    if (holidayName(t0 - D) && !hn) lines.push(`Wczoraj było święto (${holidayName(t0 - D)}).`);
    // 3) najbliższe 7 dni: święta, mosty
    const soon = [];
    for (let i = 1; i <= 7; i++) {
        const t = t0 + i * D, h = holidayName(t);
        if (h) soon.push(`${dayPl(t)}, ${dateU(t)} — ${h}${isWeekend(t) ? ' (wypada w weekend)' : ' (dzień wolny)'}`);
        else if (isBridge(t)) soon.push(`${dayPl(t)}, ${dateU(t)} — „most” (dzień roboczy między dniem wolnym a weekendem)`);
    }
    if (soon.length) lines.push(`W najbliższych 7 dniach: ${soon.join('; ')}.`);
    // 4) najbliższe dwa święta
    const next = [];
    for (let i = 1; i <= 400 && next.length < 2; i++) { const t = t0 + i * D, h = holidayName(t); if (h) next.push(`${h} (${dayPl(t)}, ${dateU(t)}; za ${i} ${plural(i, 'dzień', 'dni', 'dni')})`); }
    if (next.length) lines.push(`Najbliższe święta: ${next.join('; ')}.`);
    // 5) miesiąc / pora roku
    lines.push(`Ogólne tło (miesiąc i pora roku): ${MIESIACE[+p.month - 1]} — ${monthNote(+p.month, +p.day)}.`);
    return lines.join('\n');
}

function calendarContext(now = new Date()) {
    const p = warsawParts(now), dow = new Date(Date.UTC(+p.year, +p.month - 1, +p.day)).getUTCDay();
    const mon = new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, day: 'numeric', month: 'long' }).formatToParts(now).find(x => x.type === 'month')?.value || '';
    return {
        dzis: {
            data: new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, day: 'numeric', month: 'long', year: 'numeric' }).format(now),
            dzienTygodnia: p.weekday, godzina: `${p.hour}:${p.minute}`, dataISO: `${p.year}-${p.month}-${p.day}`, weekend: dow === 0 || dow === 6,
            dzisSlownie: `${p.weekday}, ${DZIEN_PORZ[+p.day]} ${mon}`,                  // np. „wtorek, siedemnasty listopada”
            godzinaSlownie: godzMianownik(+p.hour % 24, +p.minute)                        // np. „dwudziesta pierwsza dwadzieścia trzy”
        },
        kalendarz: calendarBlock(now)
    };
}

function withClock(s) {
    if (!s || !Array.isArray(s.przebieg) || !s.start) return s;
    const t0 = Date.parse(s.start);
    return { ...s, przebieg: s.przebieg.map(x => ({ godzina: clock(new Date(t0 + (x.min || 0) * 6e4)), ...x })) };
}

// ═══════════════ WYLICZENIA STATYSTYCZNE (w kodzie, model tylko interpretuje) ═══════════════
const round = (n, d = 0) => Number.isFinite(n) ? Math.round(n * 10 ** d) / 10 ** d : null;
const pct = (a, b) => (Number.isFinite(a) && Number.isFinite(b) && b > 0) ? Math.round((a - b) / b * 100) : null;
const band = hh => hh >= 6 && hh < 12 ? 'rano (6–11)' : hh >= 12 && hh < 15 ? 'południe (12–14)' : hh >= 15 && hh < 18 ? 'popołudnie (15–17)' : hh >= 18 && hh < 23 ? 'wieczór (18–22)' : 'noc (23–5)';

function sessionMetrics(s) {
    const h = Math.max(0.05, (s.czasTrwaniaMin || 0) / 60);
    const subs = s.noweSuby ?? s.noweSubskrypcje ?? 0;
    return {
        h, avg: s.sredniaWidzow || 0, peak: s.peakWidzow || 0,
        msgH: (s.wiadomosciNaCzacie || 0) / h, folH: (s.nowiFollowersi || 0) / h, subH: subs / h,
        chatShare: s.szac_unikalniWidzowie > 0 ? (s.unikalniCzatujacy || 0) / s.szac_unikalniWidzowie : null,
        msgPerViewerHour: (s.sredniaWidzow > 0) ? (s.wiadomosciNaCzacie || 0) / (s.sredniaWidzow * h) : null,
        folPVH: (s.sredniaWidzow > 0) ? (s.nowiFollowersi || 0) / (s.sredniaWidzow * h) : null   // followy na widza na godzinę (×100 = na 100 godzin oglądania)
    };
}
function wavg(items, key) {
    let ws = 0, vs = 0;
    for (const it of items) { const v = it.m[key]; if (v == null || !Number.isFinite(v)) continue; ws += it.w; vs += v * it.w; }
    return ws ? vs / ws : null;
}
function groupBy(list, keyFn, top = 8) {
    const map = {};
    list.forEach(it => { const k = keyFn(it); if (!k) return; (map[k] = map[k] || []).push(it); });
    return Object.entries(map).map(([k, arr]) => ({
        klucz: k, liczbaSesji: arr.length, lacznieGodzin: round(arr.reduce((a, x) => a + x.m.h, 0), 1),
        sredniaWidzow_wazona: round(wavg(arr, 'avg'), 1), sredniPeak_wazony: round(wavg(arr, 'peak'), 1),
        wiadomosciNaGodzine_wazone: round(wavg(arr, 'msgH'), 1), nowiFollowersiNaGodzine_wazone: round(wavg(arr, 'folH'), 2),
        noweSubyNaGodzine_wazone: round(wavg(arr, 'subH'), 2),
        ...(arr.length === 1 ? { uwaga: 'pojedyncza sesja' } : arr.length === 2 ? { uwaga: 'tylko dwie sesje' } : {})
    })).sort((a, b) => (b.sredniaWidzow_wazona || 0) - (a.sredniaWidzow_wazona || 0)).slice(0, top);
}

// ── Najlepsze dni, pory i godziny (liczone w kodzie; model tylko opowiada) ──
const hourOfTs = t => (+warsawParts(new Date(t)).hour) % 24;
const hh2 = h => String(h).padStart(2, '0') + ':00';
const sampleLabel = n => n <= 1 ? 'jedna sesja' : n <= 3 ? 'mała próba' : n <= 7 ? 'wstępna próba' : 'solidna próba';
const meanOf = a => a.reduce((s, x) => s + x, 0) / a.length;

// Profil jednej sesji w OKNACH 30-MINUTOWYCH (czas polski; okno 0 = 00:00–00:30 … okno 47 = 23:30–24:00):
// [{ s: okno, v: śr. widzów, w: minuty streama w oknie, age: ile godz. po starcie }].
// Aplikacja wysyła tylko [[godzina, śr. widzów], ...] ("godzinyProfil", godzina z .5 = druga połowa, np. 18.5 = 18:30) dla KAŻDEJ zapisanej sesji;
// minuty i "ile po starcie" serwer liczy sam ze startu i końca sesji. Okno musi mieć w sesji min. 12 minut streama (inaczej to skrawek).
// Starsza aplikacja nie wysyła profilu — wtedy liczymy go z "przebieg". Polska ma całkowite przesunięcie UTC, więc okna zegarowe = okna UTC.
const MIN_SLOT_MIN = 12;
// Waga sesji we WZORCACH (dni, pory, okna godzinowe): łagodny spadek, żeby 25. (najstarsza) sesja też się liczyła, tylko najmniej (0,95^24 ≈ 0,29).
// Porównanie „najnowsza sesja vs norma” i wyniki wg gier używają osobnej stałej COMPARE_DECAY (też domyślnie 0,95: każda starsza sesja ma ~5% mniejszą wagę, więc 25. sesja nadal się liczy).
// Zmienne środowiskowe PATTERN_DECAY i COMPARE_DECAY (0–1; 1 = wszystkie sesje równie ważne, mniejsza liczba = szybsze „zapominanie”).
const PATTERN_DECAY = (Number(process.env.PATTERN_DECAY) > 0 && Number(process.env.PATTERN_DECAY) <= 1) ? Number(process.env.PATTERN_DECAY) : 0.95;
const COMPARE_DECAY = (Number(process.env.COMPARE_DECAY) > 0 && Number(process.env.COMPARE_DECAY) <= 1) ? Number(process.env.COMPARE_DECAY) : 0.95;
// Rajd/hosting liczy się dopiero od takiego wzrostu widowni (osoby ponad poziom sprzed skoku). Przy większych kanałach i tak decyduje próg ×1,8 z aplikacji. Zmienna SPIKE_MIN.
const SPIKE_MIN = Number(process.env.SPIKE_MIN) || 50;
const PB_MAX = Math.max(8, Number(process.env.PRZEBIEG_LINII) || 48);   // ile linii przebiegu najnowszej sesji trafia do promptu
const slotOf = t => hourOfTs(t) * 2 + (new Date(t).getUTCMinutes() >= 30 ? 1 : 0);
const okN = x => (Number.isFinite(x) && x >= 0) ? x : null;
function hourProfile(s) {
    const t0 = Date.parse(s.start), t1 = s.koniec ? Date.parse(s.koniec) : t0 + (s.czasTrwaniaMin || 0) * 6e4;
    if (!Number.isFinite(t0) || !(t1 > t0)) return [];
    let rows = [];   // [okno, śr. widzów]
    if (Array.isArray(s.godzinyProfil) && s.godzinyProfil.length) {
        rows = s.godzinyProfil.filter(r => Array.isArray(r) && r.length >= 2 && Number.isFinite(r[0]) && Number.isFinite(r[1]) && r[1] >= 0)
            .map(r => [(((Math.round(r[0] * 2)) % 48) + 48) % 48, r[1], okN(r[2]), okN(r[3])]);   // [okno, śr. widzów, wiadomości w oknie, nowe followy w oknie]
    } else {
        const per = {};
        (s.przebieg || []).filter(x => Number.isFinite(x?.v) && Number.isFinite(x?.min)).forEach(x => { const k = slotOf(t0 + x.min * 6e4), e = per[k] = per[k] || { s: 0, n: 0, m: 0 }; e.s += x.v; e.n++; if (Number.isFinite(x.msg) && x.msg > 0) e.m += x.msg; });
        rows = Object.entries(per).map(([k, e]) => [+k, e.s / e.n, e.m, null]);
    }
    if (!rows.length) return [];
    const cov = {};
    for (let t = t0; t < t1;) {
        const nxt = Math.min(t1, (Math.floor(t / 18e5) + 1) * 18e5), min = (nxt - t) / 6e4, k = slotOf(t);
        const e = cov[k] = cov[k] || { min: 0, ageW: 0 }; e.min += min; e.ageW += ((t + nxt) / 2 - t0) / 36e5 * min;
        t = nxt;
    }
    return rows.filter(([k]) => cov[k] && cov[k].min >= MIN_SLOT_MIN).map(([k, v, m, f]) => ({ s: k, v, w: cov[k].min, age: cov[k].ageW / cov[k].min, m, f }));
}
const slotTime = k => { const m = (k % 48) * 30; return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };

function bestMoments(items) {
    // DNI TYGODNIA: wynik 0–100 = oglądalność 50% + followy/h 30% + wiadomości/h 20% (każde względem najlepszego dnia)
    const byDay = {};
    items.forEach(it => { const k = it.s.dzienTygodnia; if (k) (byDay[k] = byDay[k] || []).push(it); });
    const dayRows = Object.entries(byDay).map(([k, arr]) => ({
        dzien: k, sesji: arr.length, roznychDni: new Set(arr.map(x => warsawYmd(new Date(x.s.start)))).size,
        avg: wavg(arr, 'avg'), fol: wavg(arr, 'folH'), msg: wavg(arr, 'msgH'), fvh: wavg(arr, 'folPVH'), mvh: wavg(arr, 'msgPerViewerHour')
    }));
    const mx = key => Math.max(0, ...dayRows.map(r => r[key] || 0));
    const mA = mx('avg'), mF = mx('fol'), mM = mx('msg');
    dayRows.forEach(r => { r.wynik = Math.round((mA ? (r.avg || 0) / mA : 0) * 50 + (mF ? (r.fol || 0) / mF : 0) * 30 + (mM ? (r.msg || 0) / mM : 0) * 20); });
    const najlepszeDni = dayRows.sort((a, b) => b.wynik - a.wynik).slice(0, 3).map(r => ({
        dzien: r.dzien, wynik: r.wynik, sesji: r.sesji, roznychDni: r.roznychDni, sredniaWidzow: round(r.avg, 1),
        followyNaGodzine: round(r.fol, 2), wiadomosciNaGodzine: round(r.msg, 1), wiadomosciNaWidzaNaGodzine: round(r.mvh, 2), followyNa100GodzinOgladania: round(r.fvh * 100, 1), wiarygodnosc: sampleLabel(r.roznychDni)
    }));

    // OKNA 30-MIN I PORY: z profilu KAŻDEJ sesji — indeks = widzowie w danym oknie / średnia całego streama
    // Średnia odniesienia liczy się z CAŁEGO streama, razem z pierwszą godziną (rozpęd to cenne dane); okna z początku tylko oznaczamy ("rozped").
    const hours = {}, bands = {}; let nS = 0;
    const G = { msg: 0, min: 0, fol: 0, minF: 0, rawM: 0, rawF: 0 };   // sumy globalne (ważone) do oczekiwanego tempa czatu i followów
    items.forEach(it => {
        const prof = hourProfile(it.s), W = prof.reduce((a, r) => a + r.w, 0);
        if (!prof.length || !(W > 0)) return;
        const m = prof.reduce((a, r) => a + r.v * r.w, 0) / W; if (!(m > 0)) return;   // średnia całego streama (ważona minutami), razem z pierwszą godziną
        const perB = {};
        prof.forEach(r => {
            (hours[r.s] = hours[r.s] || []).push({ v: r.v, idx: r.v / m, w: it.w, age: r.age, early: r.age < 1, mins: r.w, msg: r.m, fol: r.f });
            if (r.m != null) { G.msg += it.w * r.m; G.min += it.w * r.w; G.rawM += r.m; }
            if (r.f != null) { G.fol += it.w * r.f; G.minF += it.w * r.w; G.rawF += r.f; }
            const bk = band(Math.floor(r.s / 2)), e = perB[bk] = perB[bk] || { s: 0, n: 0 }; e.s += r.v * r.w; e.n += r.w;
        });
        for (const [b, e] of Object.entries(perB)) (bands[b] = bands[b] || []).push({ v: e.s / e.n, idx: e.s / e.n / m, w: it.w });
        nS++;
    });
    const agg = (arr, k) => { let ws = 0, vs = 0; arr.forEach(x => { ws += x.w; vs += x[k] * x.w; }); return ws ? vs / ws : null; };
    const minS = nS >= 4 ? 2 : 1;   // przy małej historii wystarczy jedna sesja, przy większej wymagamy dwóch
    // Sygnał czatu / followów w oknie = obserwowane ÷ oczekiwane przy średnim tempie streama, ściągnięte ku 1,0 (stała K): mało wiadomości lub followów = wynik bliski 1,0,
    // więc pojedynczy follow nie robi z okna mocnego okna. Brak danych (stara aplikacja, <30 wiadomości lub <5 followów łącznie) = składnik pomijany, a jego waga idzie na resztę.
    const SIG = (arr, key, K, num, den, raw, rawMin) => {
        if (!(num > 0) || !(den > 0) || raw < rawMin) return null;
        const rows = arr.filter(x => x[key] != null);
        if (!rows.length) return null;
        const obs = rows.reduce((a, x) => a + x.w * x[key], 0), exp = num / den * rows.reduce((a, x) => a + x.w * x.mins, 0);
        return Math.min(3, (obs + K) / (exp + K));
    };
    const hourRows = Object.entries(hours).map(([k, arr]) => {
        const iV = agg(arr, 'idx'), iC = SIG(arr, 'msg', 20, G.msg, G.min, G.rawM, 30), iF = SIG(arr, 'fol', 4, G.fol, G.minF, G.rawF, 5);
        let sm = 0.5 * iV, ws = 0.5;   // indeks złożony: widownia 50%, followy 30%, czat 20% (jak wynik dnia)
        if (iC != null) { sm += 0.2 * iC; ws += 0.2; }
        if (iF != null) { sm += 0.3 * iF; ws += 0.3; }
        return { k: +k, sesji: arr.length, hit: arr.filter(x => x.idx > 1).length, widzow: agg(arr, 'v'), idxC: iC, idxF: iF, indeks: sm / ws, odStartu: agg(arr, 'age'), rozped: arr.filter(x => x.early).length / arr.length >= 0.5 };
    }).filter(r => r.sesji >= minS).sort((a, b) => a.k - b.k);
    // Próg „mocnego okna” rośnie delikatnie przy małej widowni (±1 osoba to tam szum): 3 widzów ≈ 1,23; 6 ≈ 1,16; 16 i więcej = 1,1; maks. 1,25.
    // Od 3 sesji okno musi też być ponad średnią streama w co najmniej połowie sesji (jeden wyjątkowy dzień nie robi wzorca).
    const thr = r => Math.min(1.25, Math.max(1.1, 1 + 0.4 / Math.sqrt(Math.max(1, r.widzow))));
    const hot = hourRows.filter(r => r.indeks >= thr(r) && (r.sesji < 3 || r.hit / r.sesji >= 0.5));   // tylko mocne okna; słabych nie przekazujemy
    // ciągłe przedziały mocnych okien (np. 20:00–22:30) ze średnim indeksem i liczbą sesji (minimum z okien)
    const runs = [];
    for (const r of hot) { const l = runs[runs.length - 1]; if (l && r.k === l.rows[l.rows.length - 1].k + 1) l.rows.push(r); else runs.push({ rows: [r] }); }
    const part = (rs, key) => { const v = rs.map(x => x[key]).filter(x => x != null); return v.length ? round(meanOf(v), 2) : undefined; };
    const mocnePrzedzialy = runs.map(u => { const low = u.rows.reduce((a, x) => (x.hit / x.sesji < a.hit / a.sesji ? x : a)); return ({ okres: `${slotTime(u.rows[0].k)}–${slotTime(u.rows[u.rows.length - 1].k + 1)}`, okresSlownie: okresSlownie(slotTime(u.rows[0].k), slotTime(u.rows[u.rows.length - 1].k + 1)), indeks: round(meanOf(u.rows.map(x => x.indeks)), 2), widzow: round(meanOf(u.rows.map(x => x.widzow)), 1), czatIdx: part(u.rows, 'idxC'), followyIdx: part(u.rows, 'idxF'), powyzejSredniej: `${low.hit}/${low.sesji}`, sesjiMin: Math.min(...u.rows.map(x => x.sesji)), wiarygodnosc: sampleLabel(Math.min(...u.rows.map(x => x.sesji))), _w: meanOf(u.rows.map(x => x.indeks)) }); })
        .sort((a, b) => b._w - a._w).slice(0, 3).map(({ _w, ...x }) => x);
    const najlepszePory = Object.entries(bands).map(([b, arr]) => ({
        pora: b, sesji: arr.length, sredniaWidzow: round(agg(arr, 'v'), 1), indeks: round(agg(arr, 'idx'), 2), wiarygodnosc: sampleLabel(arr.length)
    })).filter(r => r.sesji >= minS && r.indeks >= 0.9).sort((a, b) => b.sredniaWidzow - a.sredniaWidzow).slice(0, 2);   // pora wyraźnie poniżej średniej streama (np. końcówka nocna) nie jest „najlepszą porą”

    return {
        uwaga: 'Wynik dnia 0–100: widownia 50%, followy/h 30%, wiadomości/h 20%. Okna 30-min: indeks złożony (widownia 50%, followy 30%, czat 20%; brak danych = składnik pomijany), 1,0 = średnia streama; czatIdx i followyIdx to składniki (1,0 = średnia streama); powyzejSredniej = w ilu sesjach widownia okna była ponad średnią streama; rozped=true: okno z początku streama (rozpęd, dane cenne); odStartuH: godz. po starcie; sesjiZPrzebiegiem: liczba sesji z oknami.',
        najlepszeDni, najlepszePory,
        mocneOkna: [...hot].sort((a, b) => b.indeks - a.indeks).slice(0, 2).map(r => ({ okno: `${slotTime(r.k)}–${slotTime(r.k + 1)}`, oknoSlownie: okresSlownie(slotTime(r.k), slotTime(r.k + 1)), sesji: r.sesji, widzow: round(r.widzow, 1), indeks: round(r.indeks, 2), czatIdx: round(r.idxC, 2), followyIdx: round(r.idxF, 2), powyzejSredniej: `${r.hit}/${r.sesji}`, odStartuH: round(r.odStartu, 1), rozped: r.rozped || undefined, wiarygodnosc: sampleLabel(r.sesji) })),
        mocnePrzedzialy: mocnePrzedzialy.length ? mocnePrzedzialy : undefined,
        sesjiZPrzebiegiem: nS
    };
}

// Dni z kilkoma streamami (np. rano i wieczorem): gotowe porównanie na godzinę
function multiDay(items) {
    const by = {};
    items.forEach(it => { const k = warsawYmd(new Date(it.s.start)); (by[k] = by[k] || []).push(it); });
    return Object.entries(by).filter(([, a]) => a.length >= 2).sort((a, b) => b[0].localeCompare(a[0])).slice(0, 3).map(([k, a]) => {
        a.sort((x, y) => Date.parse(x.s.start) - Date.parse(y.s.start));
        return {
            data: k, dzien: a[0].s.dzienTygodnia,
            sesje: a.map(it => ({ od: it.s.godzinaStartu, gra: it.s.glownaGra, godzin: round(it.m.h, 1), czasFraza: durFraza(it.m.h * 60), sredniaWidzow: round(it.m.avg, 1), peak: it.m.peak, followyNaGodzine: round(it.m.folH, 2), wiadomosciNaGodzine: round(it.m.msgH, 1) })),
            razem: { godzin: round(a.reduce((s, x) => s + x.m.h, 0), 1), czasFraza: durFraza(a.reduce((s, x) => s + x.m.h, 0) * 60), noweFollowy: a.reduce((s, x) => s + (x.s.nowiFollowersi || 0), 0) }
        };
    });
}

// ── Followy z ostatnich 30 dni: kiedy wpadają i kiedy opłaca się streamowanie (followy na godzinę streamu) ──
const dowHourFmt = new Intl.DateTimeFormat('en-US', { timeZone: TZ, weekday: 'short', hour: 'numeric', hourCycle: 'h23' });
const WD_IDX = { Mon: 0, Tue: 1, Wed: 2, Thu: 3, Fri: 4, Sat: 5, Sun: 6 };
const DAY_PL = ['poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela'];
function dowHour(t) {
    let d = 0, h = 0;
    for (const x of dowHourFmt.formatToParts(new Date(t))) { if (x.type === 'weekday') d = WD_IDX[x.value] ?? 0; if (x.type === 'hour') h = (+x.value) % 24; }
    return [d, h];
}
function followMoments(fk, sessions) {
    const grid = fk?.siatka_dzienGodzina;   // { "dzień(0=pon)-godzina": liczba followów } z ostatnich 30 dni, czas polski
    if (!grid || typeof grid !== 'object') return null;
    const since = Date.now() - 30 * 864e5, cov = {};   // cov: minuty streamowania w danej komórce dzień×godzina
    for (const s of sessions) {
        const t0 = Date.parse(s.start), t1 = s.koniec ? Date.parse(s.koniec) : t0 + (s.czasTrwaniaMin || 0) * 6e4, a = Math.max(t0, since);
        if (!(t1 > a)) continue;
        for (let t = a + 150e3; t < t1; t += 300e3) { const [d, h] = dowHour(t), k = d + '-' + h; cov[k] = (cov[k] || 0) + 5; }
    }
    const mk = () => ({ f: 0, fS: 0, min: 0 });
    const D = Array.from({ length: 7 }, mk), H = Array.from({ length: 24 }, mk), B = {};
    const bb = h => B[band(h)] = B[band(h)] || mk();
    let total = 0;
    for (const [k, c] of Object.entries(grid)) {   // wszystkie followy (także poza streamem)
        const [d, h] = k.split('-').map(Number), n = Number(c) || 0;
        if (!(d >= 0 && d < 7 && h >= 0 && h < 24) || n <= 0) continue;
        total += n; D[d].f += n; H[h].f += n; bb(h).f += n;
    }
    for (const [k, m] of Object.entries(cov)) {   // tylko komórki, w których streamowałeś co najmniej godzinę łącznie
        if (m < 60) continue;
        const [d, h] = k.split('-').map(Number), n = Number(grid[k]) || 0;
        for (const x of [D[d], H[h], bb(h)]) { x.min += m; x.fS += n; }
    }
    const rate = x => x.min > 0 ? x.fS / (x.min / 60) : 0;
    const per = (x, label, name) => ({ [label]: name, ...(label === 'godz' ? { godzinaSlownie: godzSlownie(name) } : {}), followy: x.fS, godzinStreamu: round(x.min / 60, 1), naGodzine: round(rate(x), 2) });
    const bandNames = Object.keys(B);
    const best = (rows, minMin, n) => rows.filter(r => r.x.min >= minMin && r.x.fS > 0).sort((a, b) => rate(b.x) - rate(a.x)).slice(0, n);
    const totalBy = (rows, n) => rows.filter(r => r.x.f > 0).sort((a, b) => b.x.f - a.x.f).slice(0, n);
    const dRows = D.map((x, i) => ({ x, name: DAY_PL[i] })), hRows = H.map((x, i) => ({ x, name: hh2(i) })), bRows = bandNames.map(b => ({ x: B[b], name: b }));
    return {
        okresDni: 30, lacznie: total, wTrakcieStreamow_procent: fk.udzialFollowowWTrakcieStreamow_procent ?? undefined,
        probka: total < 15 ? 'za mała (poniżej 15 followów)' : 'wystarczająca jak na mały kanał',
        naGodzineStreamu: {
            najlepszeDni: best(dRows, 120, 3).map(r => per(r.x, 'dzien', r.name)),
            najlepszePory: best(bRows, 120, 2).map(r => per(r.x, 'pora', r.name)),
            najlepszeGodziny: best(hRows, 60, 3).map(r => per(r.x, 'godz', r.name))
        },
        najwiecejOgolem: {
            dni: totalBy(dRows, 3).map(r => ({ dzien: r.name, followy: r.x.f })),
            pory: totalBy(bRows, 2).map(r => ({ pora: r.name, followy: r.x.f })),
            godziny: totalBy(hRows, 3).map(r => ({ godz: r.name, followy: r.x.f }))
        }
    };
}

function derive(current, history, kanal) {
    const all = [...(current ? [current] : []), ...(history || [])].filter(s => s && s.czasTrwaniaMin > 0 && s.start);
    // wagi: najnowsza sesja = 1, każda starsza × 0,95 (COMPARE_DECAY / PATTERN_DECAY)
    const items = all.map((s, i) => ({ s, m: sessionMetrics(s), w: Math.pow(COMPARE_DECAY, i), wp: Math.pow(PATTERN_DECAY, i) }));   // w: porównania i gry; wp: wzorce (dni, pory, okna)
    const histItems = items.filter(it => it.s !== current);
    const hourOf = s => parseInt(String(s.godzinaStartu || '').slice(0, 2), 10);
    const out = { liczbaSesjiWHistorii: histItems.length };
    out.wedlugGier = groupBy(items, it => it.s.glownaGra);
    out.najlepszeMomenty = bestMoments(items.map(it => ({ ...it, w: it.wp })));   // wzorce liczone łagodniejszymi wagami
    const md = multiDay(items); if (md.length) out.dniZWieluStreamami = md;
    const fmo = followMoments(kanal?.followyKiedy, all); if (fmo) out.followyMomenty = fmo;
    if (current && histItems.length) {
        const cm = sessionMetrics(current);
        const base = k => wavg(histItems, k);
        const mk = arr => arr.length ? {
            sredniaWidzow: round(wavg(arr, 'avg'), 1), peak: round(wavg(arr, 'peak'), 1), wiadomosciNaGodzine: round(wavg(arr, 'msgH'), 1),
            nowiFollowersiNaGodzine: round(wavg(arr, 'folH'), 2), liczbaSesjiWBazie: arr.length
        } : null;
        out.porownanieNajnowszejSesji = {
            uwaga: 'Wiadomości i followersi na godzinę streama; średnia widzów i peak bezpośrednio (uwzględnij różnicę długości).' + (current.trwaJeszcze ? ' Sesja jeszcze trwa, więc porównanie jest wstępne.' : ''),
            czasTrwaniaNajnowszejH: round(cm.h, 2), sredniCzasTrwaniaHistoriiH: round(base('h'), 2),
            odchylenieProcent_wzgledemWazonejHistorii: {
                sredniaWidzow: pct(cm.avg, base('avg')), peak: pct(cm.peak, base('peak')),
                wiadomosciNaGodzine: pct(cm.msgH, base('msgH')), nowiFollowersiNaGodzine: pct(cm.folH, base('folH'))
            },
            historiaWazona: mk(histItems),
            tenSamDzienTygodnia: mk(histItems.filter(it => it.s.dzienTygodnia === current.dzienTygodnia)),
            taSamaGra: current.glownaGra ? mk(histItems.filter(it => it.s.glownaGra === current.glownaGra)) : null,
            takaSamaPoraStartu: mk(histItems.filter(it => band(hourOf(it.s) || 0) === band(hourOf(current) || 0))),
            udzialCzatujacychWSzacowanychUnikalnych: { najnowsza: round(cm.chatShare, 2), historia: round(base('chatShare'), 2) },
            wiadomosciNaWidzaNaGodzine: { najnowsza: round(cm.msgPerViewerHour, 2), historia: round(base('msgPerViewerHour'), 2) }
        };
    }
    return out;
}

// Notatka na początek wypowiedzi. Czas „temu” zawsze jako gotowa fraza (temu()), a streamy z dzisiejszą datą
// (bieżący + z historii) są wymienione razem, żeby model widział, że to ten sam dzień.
function lastSessionNote(current, cal, live, history = []) {
    if (!current) return 'Brak zapisanej historii sesji w widżecie.';
    const now = new Date(), startD = new Date(current.start);
    const endOf = s => s.koniec ? new Date(s.koniec) : new Date(new Date(s.start).getTime() + (s.czasTrwaniaMin || 0) * 6e4);
    const slot = s => `${okresSlownie(s.godzinaStartu || clock(new Date(s.start)), clock(endOf(s)))}, trwał ${durFraza(sessMin(s))}${s.glownaGra ? ` (${s.glownaGra})` : ''}`;
    const sumTxt = list => `Łączny czas streamowania dziś: ${durFraza(list.reduce((a, x) => a + sessMin(x), 0))}.`;
    const byStart = (a, b) => Date.parse(a.start) - Date.parse(b.start);
    const isToday = s => s && s.start && dayDiff(now, new Date(s.start)) === 0;
    const seen = new Set();
    const todayAll = [current, ...(history || [])].filter(s => {
        if (!isToday(s) || seen.has(s.start)) return false;
        seen.add(s.start); return true;
    }).sort(byStart);
    const word = n => ['', 'jeden', 'dwa', 'trzy', 'cztery'][n] || String(n);

    if (live) {
        const others = todayAll.filter(s => s.start !== current.start);
        const head = `Stream trwa teraz. Start: ${current.dzienTygodnia}, o ${godzSlownie(current.godzinaStartu)} (${temu(startD.getTime(), now)}).`;
        if (!others.length) return head;
        const last = others[others.length - 1];
        return `${head} Dziś był już wcześniej ${others.length === 1 ? 'jeden stream' : word(others.length) + ' streamy'}: ${others.map(slot).join(', ')}. Ostatni z nich skończył się ${temu(endOf(last).getTime(), now)}. Czasy trwania podaj wyłącznie tak, jak tu zapisano.`;
    }

    const endD = endOf(current), dd = dayDiff(now, endD);
    const todayList = [...todayAll];
    if (dd === 0 && !todayList.some(s => s.start === current.start)) todayList.push(current);   // stream zaczęty wczoraj, skończony dziś
    todayList.sort(byStart);
    if (todayList.length >= 2) {
        const lastEnd = Math.max(...todayList.map(s => endOf(s).getTime()));
        return `Dziś były już ${word(todayList.length)} streamy: ${todayList.map(slot).join(', ')}. ${sumTxt(todayList)} Ostatni skończył się ${temu(lastEnd, now)}. Czasy trwania i sumę podaj wyłącznie tak, jak tu zapisano — nie dodawaj ich sama.`;
    }
    if (todayList.length === 1) return `Dziś był już stream: start o ${godzSlownie(current.godzinaStartu)}, koniec około ${godzSlownie(clock(endD))}, trwał ${durFraza(sessMin(current))}, skończył się ${temu(endD.getTime(), now)}.`;
    return `Dziś nie było jeszcze streama. Ostatni skończył się ${temu(endD.getTime(), now)} (${current.dzienTygodnia}, start o ${godzSlownie(current.godzinaStartu)}, koniec około ${godzSlownie(clock(endD))}).`;
}

// ═══════════════ BADANIE SIECI (z pamięcią podręczną) ═══════════════
// Kalendarz (zwykłe święta, dni wolne, długie weekendy, pora roku) liczy serwer, więc sieć sprawdza tylko to, czego kod nie wie.
// Jedno zapytanie, maks. 3 wyszukiwania, pamięć podręczna 12 h (klucz: doba + zestaw gier). Analiza końcowa nie używa wyszukiwarki.
// KOLEJNOŚĆ WAŻNOŚCI (od najważniejszej): 1) Polska: wydarzenia + sport (w tym dodatkowe dni wolne, żałoba, zmiany w kalendarzu świąt),
// 2) Twitch: wydarzenia + najpopularniejsze gry, 3) świat gier: premiery i głośne wydarzenia, 4) gra streamera: patche, oglądalność.
// Do analizy trafia tylko to, co udało się potwierdzić; sekcje bez wyniku są pomijane.
function researchPrompt(cal, games) {
    return `Dziś jest ${cal.dzis.dzienTygodnia}, ${cal.dzis.data}, godz. ${cal.dzis.godzina} (czas polski). Zbierz w sieci informacje dla DZISIAJ i NAJBLIŻSZYCH 7 DNI, w poniższej kolejności ważności, używając maksymalnie 3 wyszukiwań. Zwykłe święta ustawowe, weekendy, długie weekendy i pory roku są już policzone — ich NIE sprawdzaj.
1. POLSKA (najważniejsze): duże wydarzenia w Polsce na dziś lub nadchodzący tydzień; w tym dodatkowe dni wolne lub zmiany w kalendarzu świąt, jeśli są. Osobno SPORT: duże wydarzenia sportowe w Polsce na dziś lub nadchodzący tydzień (mecze, turnieje, gale).
2. TWITCH: największe wydarzenia na Twitchu oraz najpopularniejsze gry na polskim Twitchu teraz.
3. GRY: ważne premiery i głośne wydarzenia ze świata gier.
${games.length ? `4. GRA STREAMERA (na końcu): ${games.join('; ')} — patche, aktualizacje, eventy i dropy; ilu widzów w Polsce ją ogląda i ilu streamuje.\n` : ''}Przy każdym wydarzeniu podaj dzień i porę lub godzinę, jeśli są znane. Szukaj od punktu 1; jeśli zabraknie wyszukiwań, niższe punkty pomiń. Odpowiedz krótkimi punktami „POLSKA”, „SPORT”, „TWITCH”, „GRY”${games.length ? ', „GRA”' : ''} (łącznie ok. 150 słów). Wpisz tylko to, co udało się potwierdzić; sekcji bez potwierdzonych faktów w ogóle nie wypisuj (żadnych „brak danych”). Nie zmyślaj liczb. Bez nazw firm, portali i serwisów (wyjątek: gry, studia, wydawcy, Twitch) i bez linków.`;
}
function pickGames(body) {
    const seen = [], add = g => { if (g && typeof g === 'string' && !seen.some(x => x.toLowerCase() === g.toLowerCase())) seen.push(g.trim()); };
    const cur = body.currentSession;
    add(body.teraz?.gra); add(cur?.glownaGra); (cur?.gry || []).slice(0, 2).forEach(x => add(x.nazwa));
    (body.sessionsHistory || []).slice(0, 2).forEach(s => add(s.glownaGra));
    return seen.slice(0, 3);
}

// Ostatnie udane badanie sieci: używane TYLKO gdy świeże się nie uda (np. limit Gemini), wyłącznie z tej samej doby i do 12 h.
// Potem jest kasowane (sprawdzanie co 10 min), więc stare dane nie leżą w nieskończoność.
let lastNet = null; // { date, gk, text, sources, at }
setInterval(() => {
    if (lastNet && (Date.now() - lastNet.at > 12 * 3600e3 || lastNet.date !== calendarContext().dzis.dataISO)) lastNet = null;
}, 10 * 60e3).unref();

// Wynik badania zapamiętany na 12 godzin (klucz: data + zestaw gier)
async function research(body, cal, mode) {
    // Badanie sieci jest tylko dodatkiem: cokolwiek się tu stanie (też błąd w kodzie), analiza leci dalej bez notatek z sieci.
    let date = null, gk = '';
    try {
        date = cal.dzis.dataISO;
        const games = pickGames(body);
        gk = games.map(x => x.toLowerCase()).sort().join('+');
        const key = `net|v6|${date}|${gk}`;   // ważne 12 h w ciągu tej samej doby, osobno dla danego zestawu gier
        const r = await cacheMain.memo(key, TTL.games, async () => {
            const x = await genSeq(researchPrompt(cal, games), true, RESEARCH_MS, 'low', RESEARCH_MODELS, 4, Date.now() + RESEARCH_BUDGET_MS, x => !!String(x?.text || '').trim());
            lastNet = { date, gk, text: x.text?.trim() || '', sources: x.sources || [], at: Date.now() };
            return x;
        });
        return { net: r.text?.trim() || null, sources: r.sources || [] };
    } catch (e) {
        console.error('Badanie sieci nieudane:', e.message);
        try {
            if (lastNet && lastNet.text && date && lastNet.date === date && Date.now() - lastNet.at < 12 * 3600e3) {
                const h = Math.max(1, Math.round((Date.now() - lastNet.at) / 36e5));
                console.log('Używam ostatnich zapamiętanych notatek z sieci (sprzed ' + h + ' h)');
                return { net: lastNet.text + `\n(Uwaga: te notatki z sieci pochodzą sprzed ok. ${h} godz., mogły się zmienić${lastNet.gk !== gk ? ' i część o grach może dotyczyć innych gier' : ''}.)`, sources: lastNet.sources };
            }
        } catch (e2) { console.error('Notatki zapasowe z sieci:', e2.message); }
        return { net: null, sources: [] };
    }
}

// ═══════════════ HISTORIA JAKO ZWARTA TABELA (zamiast JSON-a: ok. 3–4× mniej tokenów, a każda sesja ma dzień, godziny i grę) ═══════════════
const DOW3 = { poniedziałek: 'pon', wtorek: 'wt', środa: 'śro', czwartek: 'czw', piątek: 'pią', sobota: 'sob', niedziela: 'nie' };
const cellTxt = v => (v === undefined || v === null || v === '') ? '-' : String(v).replace(/\|/g, '/');
function sessionRow(s) {
    const st = new Date(s.start), en = s.koniec ? new Date(s.koniec) : new Date(st.getTime() + (s.czasTrwaniaMin || 0) * 6e4);
    const tyt = s.tytul ? String(s.tytul).slice(0, 50) : '';   // jeden tytuł na sesję (ten, który trwał najdłużej)
    return [warsawYmd(st), DOW3[s.dzienTygodnia] || s.dzienTygodnia, `${s.godzinaStartu || clock(st)}–${s.godzinaKonca || clock(en)}`, s.czasTrwaniaMin, s.glownaGra || '-',
        s.sredniaWidzow, s.peakWidzow, s.wiadomosciNaCzacie, s.unikalniCzatujacy, s.nowiFollowersi, s.noweSuby ?? s.noweSubskrypcje,
        s.szac_unikalniWidzowie, s.szac_sredniCzasOgladaniaMin, tyt, Array.isArray(s.skokiWidzow) ? s.skokiWidzow.length : undefined].map(cellTxt).join('|');
}
const historyTable = h => (h || []).length
    ? `Najnowsze pierwsze, jedna sesja = jedna linia, "-" = brak danych. Kolumny: data|dzień|od–do|czas w min|gra główna|średnia widzów|peak|wiadomości na czacie|piszący|nowi followersi|nowe suby|szac. unikalni widzowie|szac. średni czas oglądania w min|tytuł|liczba skoków widowni\n${h.map(sessionRow).join('\n')}`
    : 'brak.';

// ═══════════════ NOWA PACZKA Z APLIKACJI (bloki) → układ wewnętrzny ═══════════════
// Nowa paczka: { czas, najnowszaSesja (z polem naZywo), historia: { sesje }, statystyki14dni, kanal }.
// Stara paczka (currentSession / sessionsHistory / teraz) nadal działa. Wynik zawsze ma pola: currentSession, sessionsHistory, teraz, statystyki.
function normalizeBody(b) {
    if (!b || typeof b !== 'object') return {};
    if (!b.najnowszaSesja && !b.historia && !b.czas && !b.statystyki14dni) return b;
    const o = { ...b }, cs0 = b.najnowszaSesja || null;
    const { opis: _o, naZywo, ...cs } = cs0 || {};
    o.currentSession = cs0 ? cs : (b.currentSession ?? null);
    o.sessionsHistory = Array.isArray(b.historia) ? b.historia : Array.isArray(b.historia?.sesje) ? b.historia.sesje : (b.sessionsHistory || []);
    const cz = b.czas || {};
    o.teraz = { iso: cz.iso, sesjaAktywna: cz.sesjaAktywna, gra: cz.gra ?? null, tytul: cz.tytul ?? null, aktualnie: naZywo || null };
    o.statystyki = b.statystyki14dni || null;
    return o;
}
// Porządki po obliczeniach: jeden tytuł i jedna gra na sesję; przebieg minutowy i lista gier tylko dla najnowszej sesji (stare aplikacje wysyłały więcej)
function tidySession(s, latest) {
    if (!s) return s;
    const { tytuly, gra, przebieg, gry, ...r } = s, o = { ...r };
    if (!o.tytul && Array.isArray(tytuly) && tytuly.length) o.tytul = String(tytuly[tytuly.length - 1]);
    if (latest) { if (przebieg) o.przebieg = przebieg; if (Array.isArray(gry) && gry.length) o.gry = gry; }
    return o;
}

// ═══════════════ STATYSTYKI 14 DNI (wyniki z karty STATYSTYKI w aplikacji; serwer dopisuje tylko wniosek) ═══════════════
const ST_LAB = {
    sredniaWidzow: ['średnia widzów na sesję', 0], peak: ['szczyt widzów (średnio na sesję)', 0], przewineloSie: ['szac. unikalni widzowie na sesję', 0],
    sredniCzasOgladaniaMin: ['średni czas oglądania w minutach', 0], osobNaCzacie: ['osób piszących na czacie na sesję', 0], wiadomosci: ['wiadomości na czacie na sesję', 0],
    widzowieXGodziny: ['widzowie × godziny streama (średnio na sesję)', 1], wiadomosciNa15Min: ['wiadomości na 15 minut', 1], procentWidzowPiszacych: ['ile procent widzów pisze', 1],
    wiadomosciNaCzatujacego: ['wiadomości na jedną piszącą osobę', 1], procentStreamaNaWidza: ['ile procent streama ogląda przeciętny widz', 1],
    noweFollowyNaDzien: ['nowe followy na dzień', 1], procentWidzowZostajeFollowerem: ['ile procent widzów zostaje followerem', 1]
};
const ST_REC = { sredniaWidzow: 'najwyższa średnia widzów', peak: 'najwyższy szczyt widzów', wiadomosci: 'najwięcej wiadomości', osobNaCzacie: 'najwięcej osób na czacie', noweFollowy: 'najwięcej nowych followów' };
const numPl = (n, d) => String(round(n, d)).replace('.', ',');
// Zmiana względna jak na karcie: próg 3% (poniżej = bez zmian); zero wcześniej i więcej teraz = „nowe”
function statChange(a, b) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    if (Math.abs(b) < 1e-9) return Math.abs(a) < 1e-9 ? { p: 0, txt: '0%', w: 'bez zmian' } : { p: Infinity, txt: 'od zera', w: 'rośnie' };
    const p = Math.round((a - b) / Math.abs(b) * 100);
    return { p, txt: (p > 0 ? '+' : p < 0 ? '-' : '') + Math.min(999, Math.abs(p)) + '%', w: p >= 3 ? 'rośnie' : p <= -3 ? 'maleje' : 'bez zmian' };
}
const agoTxt = n => n <= 0 ? 'dziś' : n === 1 ? 'wczoraj' : `${n} dni temu`;
function statsBlock(st, now = new Date()) {
    if (!st || typeof st !== 'object') return 'brak (aplikacja nie przysłała statystyk).';
    const o = st.okno || {}, n = o.ostatnie14Dni || {}, w = o.dni15do30 || {};
    const head = `Okno A: ostatnie 14 dni (${n.od || '?'} – ${n.do || '?'}), sesji: ${n.sesji ?? '?'}. Okno B: dni 15–30 wcześniej (${w.od || '?'} – ${w.do || '?'}), sesji: ${w.sesji ?? '?'}.`;
    const out = [head];
    const rows = [];
    const add = (grp) => Object.entries(grp || {}).forEach(([k, v]) => {
        const lab = ST_LAB[k]; if (!lab || !Array.isArray(v)) return;
        const c = statChange(v[0], v[1]);
        rows.push({ lab: lab[0], a: v[0], b: v[1], d: lab[1], c });
    });
    add(st.srednieNaSesje); add(st.wskazniki);
    if (st.porownanieMozliwe === false || !rows.some(r => r.c)) {
        out.push('Porównanie niemożliwe: w jednym z okresów nie ma jeszcze żadnej zakończonej sesji. Nie porównuj okresów, możesz tylko wspomnieć o rekordach.');
    } else {
        out.push('Kolumny: wskaźnik|ostatnie 14 dni|dni 15–30 wcześniej|zmiana względna|wniosek (zmiana poniżej 3% = bez zmian)');
        rows.forEach(r => out.push([r.lab, r.a == null ? '-' : numPl(r.a, r.d), r.b == null ? '-' : numPl(r.b, r.d), r.c ? r.c.txt : '-', r.c ? r.c.w : '-'].join('|')));
        const up = rows.filter(r => r.c && r.c.w === 'rośnie').sort((x, y) => y.c.p - x.c.p).slice(0, 3).map(r => `${r.lab} (${r.c.txt})`);
        const dn = rows.filter(r => r.c && r.c.w === 'maleje').sort((x, y) => x.c.p - y.c.p).slice(0, 3).map(r => `${r.lab} (${r.c.txt})`);
        out.push(`Na plus: ${Number.isFinite(st.naPlus) ? st.naPlus : rows.filter(r => r.c && r.c.w === 'rośnie').length}, na minus: ${Number.isFinite(st.naMinus) ? st.naMinus : rows.filter(r => r.c && r.c.w === 'maleje').length} (wskaźników).`);
        if (up.length) out.push('Największe plusy: ' + up.join('; ') + '.');
        if (dn.length) out.push('Największe minusy: ' + dn.join('; ') + '.');
        if (Math.min(n.sesji ?? 9, w.sesji ?? 9) <= 2) out.push('Uwaga: w jednym z okien są najwyżej dwie sesje — wnioski ostrożnie.');
    }
    const rec = (st.rekordy30Dni || []).filter(r => ST_REC[r.k] && Number.isFinite(r.wartosc)).map(r => {
        const ago = Math.round((Date.parse(warsawYmd(now) + 'T00:00:00Z') - Date.parse(r.data + 'T00:00:00Z')) / 864e5);
        return `${ST_REC[r.k]}: ${numPl(r.wartosc, 0)} (${Number.isFinite(ago) ? agoTxt(ago) : r.data})`;
    });
    if (rec.length) out.push('Rekordy z 30 dni: ' + rec.join('; ') + '.');
    return out.join('\n');
}

// ═══════════════ PROMPT ANALIZY ═══════════════
function buildPrompt(body, cal, derived, ctx) {
    const { channelName = 'Dolaricious', sessionsHistory = [], currentSession = null, live = false, kanal = {}, teraz = {}, wariantCTA = 'obserwowanie', statystyki = null } = body;
    const note = lastSessionNote(currentSession, cal, live, sessionsHistory);
    const ctaTxt = wariantCTA === 'obserwowanie_i_suby'
        ? 'subtelnie zachęć widzów do zaobserwowania kanału ORAZ do zostawienia suba (jedno–dwa miękkie zdania)'
        : 'subtelnie zachęć widzów wyłącznie do zaobserwowania kanału (jedno miękkie zdanie, bez wzmianki o subskrypcji)';

    const rating = `OCENA: pod koniec wystaw ogólną ocenę najnowszej sesji w skali 1–10 (liczba całkowita, powiedziana słowem) i krótko ją uzasadnij. Oceniaj wyłącznie wyniki najnowszej sesji na tle sesji historycznych; wyniki kanału z ostatnich dni nie wpływają na ocenę. Ocena ma zgadzać się z faktami: wyniki wyraźnie poniżej normy lub długie okresy prawie bez widowni to ocena niska (1–4) i uczciwy opis, co zawiodło; przeciętnie 5–6; dobrze 7–8; wyjątkowo 9–10. Nie szukaj plusów na siłę i nie zawyżaj oceny z sympatii.${live ? ' Stream trwa, więc oceń dotychczasowy przebieg i zaznacz, że to ocena „na razie”.' : ''}`;

    const modeBlock = live ? `
═══ TRYB: LIVE (stream trwa, słyszą widzowie — audycja radiowa) ═══
- Początek jak w radiu: powitanie, dzisiejszy dzień i godzina (gotowe frazy z bloku CZAS, przepisz je), a płynnie zaraz po tym zdanie, że to analiza kanału ${channelName}. Potem od razu to, co dzieje się dziś.
- Styl: ciepła, żywa prowadząca audycji, lekko komentatorska; krótkie, rytmiczne zdania; mała historia („dzieje się to, bo…”) zamiast listy wyników; wolno zwracać się do słuchaczy. Liczby to akcenty, nie szkielet; w całości najwyżej ok. dziesięciu.
- Obowiązkowo: aktualna liczba widzów i średnia sesji (razem, w jednym zdaniu), wiadomości i piszący (w drugim), nowe followy i suby tej sesji (jeśli są), dzień i godzina startu, gra i tytuł (krótkim zwrotem), czy widownia i czat rosną, maleją czy są stabilne (blok STAN NA ŻYWO). Wcześniejszy dzisiejszy stream — jedno zdanie, z gotową frazą czasu.
- Podział czasu: ok. 45% bieżący stream na tle poprzedniej sesji i historii; 25% wzorce i godziny; 15% statystyki 14 dni; 15% wydarzenia ze świata (z kontekstu z sieci, od najważniejszych: Polska i sport, Twitch, gry, na końcu patche i oglądalność gry) i zaproszenie na koniec.
- ${rating}
- Zakończ CTA: ${ctaTxt}. Nienachalnie, bez błagania, tym samym tonem co reszta.
- Długość: 250–300 słów (ok. 2 minuty mówienia).` : `
═══ TRYB: OFFLINE (stream zakończony, słyszy tylko streamer — koleżeńska rozmowa o danych) ═══
- Zwracaj się do streamera bezpośrednio, na „Ty”. Rzeczowo, ale swobodnie, jak koleżanka, która zna się na liczbach i mówi do niego wprost. Bez CTA do widzów.
- Początek: powitanie, dzisiejszy dzień i godzina (gotowe frazy z bloku CZAS), płynnie zdanie, że to analiza kanału ${channelName}, i od razu notatka o ostatniej sesji.
- Ok. 50%: dzisiejsze streamy łącznie (jeśli było ich kilka — czasy wg gotowych fraz) i najnowsza zakończona sesja: jak poszła na tle historii (z uwzględnieniem długości) i dlaczego. Gdy sesji jest mało, skup się na danych kanału: followersi, tempo przyrostu w 30 dni, suby.
- Pozostałe 50%: wzorce (najlepsze dni, pory i godziny, gry, przyczyny), statystyki 14 dni, kalendarz oraz wydarzenia z kontekstu z sieci (od najważniejszych: Polska i sport, Twitch, gry; na końcu patche i oglądalność gry, w którą gra, ze stosunkiem widzów do streamujących).
- Każdy akapit to jeden wątek: wniosek, potem jedna–dwie liczby i przyczyna. W całości najwyżej ok. dwunastu liczb.
- ${rating}
- Długość: ok. 250–300 słów (do 2 minut mówienia).`;

    // Przebieg najnowszej sesji jako zwarta tabela (JSON z powtarzanymi nazwami pól kosztował ok. 2× więcej tokenów).
    // Do modelu nie idą: surowy profil godzinowy (godziny liczy kod) i ISO-koniec (jest godzinaKonca).
    const { przebieg: pbRaw, godzinyProfil: _gp, koniec: _ko, ...csRest } = currentSession || {};
    if (Array.isArray(csRest.gry) && csRest.gry.length < 2) delete csRest.gry;
    // Domyślnie CAŁY przebieg (do 48 punktów). Zmienna środowiskowa PRZEBIEG_LINII (np. 24) łączy sąsiednie punkty, jeśli kiedyś trzeba zaoszczędzić tokeny.
    const thinPb = pb => {
        if (pb.length <= PB_MAX) return pb;
        const k = Math.ceil(pb.length / PB_MAX), out = [];
        for (let i = 0; i < pb.length; i += k) {
            const g = pb.slice(i, i + k), vs = g.map(x => x.v).filter(Number.isFinite);
            out.push({ godzina: g[0].godzina, min: g[0].min, v: vs.length ? Math.round(meanOf(vs)) : undefined, msg: g.reduce((a, x) => a + (Number.isFinite(x.msg) ? x.msg : 0), 0), cz: g[g.length - 1].cz });
        }
        return out;
    };
    const przebiegTxt = Array.isArray(pbRaw) && pbRaw.length
        ? `\nPRZEBIEG NAJNOWSZEJ SESJI (jedna linia = jeden odcinek; kolumny: godzina|minuta od startu|śr. widzów|wiadomości w odcinku|skumulowani piszący):\n${thinPb(pbRaw).map(x => [x.godzina, x.min, x.v, x.msg, x.cz].map(cellTxt).join('|')).join('\n')}\n`
        : '';
    const liveBlock = live && teraz && teraz.aktualnie
        ? `\nSTAN NA ŻYWO (słupki = ostatnie próbki, najstarsza pierwsza; trendy policzone w aplikacji). Gra teraz: ${teraz.gra || '-'}; tytuł teraz: ${teraz.tytul || '-'}.\n${J(teraz.aktualnie)}\n`
        : '';
    const netBlock = ctx.net
        ? ctx.net
        : 'brak danych z sieci — nie wspominaj o tym, pomiń wątki gier, patchy i wydarzeń i niczego nie zmyślaj.';
    const kanalOut = derived && derived.followyMomenty ? { ...kanal, followyKiedy: undefined } : kanal;

    return `Piszesz tekst do przeczytania na głos po polsku o kanale Twitch streamera ${channelName} (polski rynek). Czyta go kobiecy głos, więc o sobie mów wyłącznie w formach żeńskich, nigdy w męskich. Nie przedstawiaj się z funkcji (żadnego „jestem analityczką” ani „analitykiem”) — prowadź płynnie, jak prezenterka. Korzystasz wyłącznie z danych poniżej: niczego nie wyszukujesz i nie zmyślasz.

═══ ZAKAZY ═══
1. Nie czytaj i nie wymieniaj nicków ani nazw użytkowników (poza ${channelName}). Żadnych kwot ani pieniędzy z subskrypcji, bitsów itp. Liczby followersów i subów wolno podawać.
2. Nie wymieniaj firm, portali, serwisów, narzędzi, sklepów, marek ani produktów. Wolno: tytuły gier, studia i wydawców gier oraz sam Twitch. Źródła opisuj ogólnie.
3. Nie doradzaj zmiany gry, dnia ani pory streamowania i nie mów, kiedy streamer powinien być online. Mocne dni i godziny podawaj wyłącznie jako obserwację. Wolno: opisywać zmiany aktywności w trakcie streama z prawdopodobnymi przyczynami, mówić o stosunku streamujących do widzów i o wydarzeniach — bez poleceń.
4. Bez marudzenia, narzekania i pouczania.

═══ PRAWDA I TON ═══
5. Ton pozytywny, ale prawdziwy: mów o tym, co poszło dobrze, i o tym, co poszło słabo. Nie szukaj plusów na siłę i nie ukrywaj minusów.
6. Nie zmyślaj liczb. Odróżniaj twarde dane, szacunki (pola „szac_”) i hipotezy; czego nie da się sprawdzić, powiedz wprost. Grupy z polem „uwaga” (jedna lub dwie sesje) to ciekawostka, nie wzorzec. Przy małej próbie („wiarygodnosc”) mów ostrożnie, ale nadal wskaż najlepsze.
7. Mów do polskich widzów o polskim Twitchu. Wydarzenia wspominaj tylko z bloku KONTEKST Z SIECI (dzisiejsze, przyszłe lub z ostatnich 7 dni) oraz z KALENDARZA; niepotwierdzonego nie dodawaj. Blok sieci jest ułożony od najważniejszego: Polska i sport, potem Twitch, potem gry, na końcu patche i oglądalność gry streamera. Mów o tym, co w nim jest, zaczynając od najważniejszego; o tym, czego w bloku nie ma, nie wspominaj i nie mów, że czegoś nie udało się sprawdzić.
8. Wydarzenia i pora jako możliwe przyczyny (hipotezy, nie fakty): wydarzenie z KONTEKSTU Z SIECI lub z KALENDARZA rozważ jako przyczynę zmiany widowni, jeśli jego pora pasuje do godzin streama z danych. Wydarzenie sportowe lub inne masowe wieczorem zwykle odciąga widzów (mniej widzów); dzień wolny i święto całodniowe zwykle zwiększają widownię, ale i liczbę streamujących, czyli konkurencję; piątkowy wieczór i weekend działają podobnie; duża premiera lub głośne wydarzenie innej gry niż streamowana odciąga widownię na inny tytuł; patch, event lub dropy w streamowanej grze mogą widownię zwiększyć. Powiąż to z kierunkiem zmiany w danych; gdy wyniki nie pokazują efektu, nie wymyślaj go. Opis miesiąca i pory roku to ogólne tło, nie wydarzenie: nie podawaj go jako newsa, używaj go tylko jako możliwego wyjaśnienia. Bez danych z sieci opieraj się na kalendarzu i porze.

═══ LICZBY I CZAS ═══
9. Słuchacz słyszy tekst raz. Najpierw wniosek własnymi słowami, potem liczba, która go potwierdza, z jej znaczeniem. Najwyżej jedna–dwie liczby w zdaniu, nigdy kilka procentów pod rząd. Zaokrąglaj jak w mowie; dokładnie tylko czas trwania i godziny. Nie czytaj nazw pól, indeksów ani współczynników. Wybierz to, co najważniejsze, resztę pomiń.
10. Czas, który upłynął, oraz czas trwania streamów i ich sumę podawaj wyłącznie gotowymi frazami z danych („czasFraza”, „temu”, „Łączny czas streamowania dziś”, „trwał…”). Nie przeliczaj, nie dodawaj, nie zaokrąglaj; gdy gotowej frazy brak, nie podawaj sumy.
11. Godziny: system 24-godzinny, słownie, bez przesuwania na 12-godzinny i bez „rano”, „po południu”, „w nocy”. Pola „okresSlownie”, „oknoSlownie”, „godzinaSlownie” przepisuj bez zmian. Nie zmieniaj, nie przesuwaj i nie zgaduj godzin.
12. Najnowszy stream porównuj do historii jako grupy, nie do pojedynczych dawnych sesji (także gdy „dniZWieluStreamami” dotyczy innego dnia). Wyjątek: inne streamy z TEGO SAMEGO DNIA — oceń dzień łącznie, potem sesje osobno wartościami na godzinę; różnicę tłumacz porą i przerwą.

═══ CO OMÓWIĆ ═══
A. Najlepsze momenty („najlepszeMomenty”, nie licz sama): maks. 2 dni tygodnia, maks. 1 pora dnia, godziny z „mocneOkna” i „mocnePrzedzialy” (przedziały czytaj z „okresSlownie”). Poprzyj jedną–dwiema liczbami; bez danych niczego nie wskazuj. Dni i godziny to osobne wnioski: nigdy nie łącz ich w jedną kombinację dnia i godziny. Przy małej widowni, ale wysokich wskaźnikach zaangażowania (czatIdx, followyIdx, wiadomosciNaWidzaNaGodzine, followyNa100GodzinOgladania) powiedz, że to mała, ale zaangażowana widownia. Możesz podać, z ilu sesji liczone są godziny („sesjiZPrzebiegiem”); „odStartuH” to liczba godzin po starcie streama. Szukaj przyczyn: weekend, święto, pora roku, wydarzenia, konkurencja.
B. Wzorce w streamie: z przebiegu najnowszej sesji i ze STANU NA ŻYWO wskaż wzrosty i spadki widowni i czatu z godziną i prawdopodobną przyczyną. Pierwsza godzina („rozped”) to rozpęd: niższa widownia na początku jest normalna.
C. Gry: nowsze mają większą wagę (policzone). Użyj kontekstu o oglądalności, stosunku widzów do streamujących, patchach i eventach; gdy go brak, pomiń.
D. Tytuł i gra a wyniki: przy „drop” lub „drops” w tytule sprawdź, czy wzrost widowni przełożył się na piszących. Więcej widzów przy czacie bez zmian to widzowie głównie „po nagrody”; rosną oba — stream faktycznie trzyma.
E. Korelacje i rajdy: więcej widzów i wiadomości to dobry znak; mało widzów, ale długo obecnych to stabilna publiczność; mało widzów i pisze ta sama grupa to własne grono. Wskaż, która sytuacja wygląda na prawdziwą. „skokiWidzow” to prawdopodobnie rajd lub hosting (hipoteza); powiedz też, jak wygląda widownia bazowa bez skoku.
F. Czas trwania: nie porównuj sesji 1:1, używaj wartości na godzinę z „porownanieNajnowszejSesji” i wspominaj o różnicy długości. Szukaj przyczyn odchyleń, nie samego procentu.
G. Followy w czasie („followyMomenty”, ostatnie 30 dni, nigdy 60): maks. 3 dni, 2 pory, 3 godziny, z jedną–dwiema liczbami i przyczynami; przy za małej próbie („probka”) powiedz, że to za mało na wnioski.
H. Tempo rozwoju: followersi w 30 dni i wzrost procentowy, saldo po unfollowach, suby (nowe i wygasłe), data ostatniego followa i suba. Nazwij dobry okres, jeśli rosną; zaznacz, że część danych subów jest przybliżona.
I. Statystyki 14 dni: wybierz kilka najważniejszych zmian z bloku STATYSTYKI (plusy i minusy) z przyczyną, jeśli widać; nie czytaj tabeli. To porównanie okresów, nie myl go z porównaniem najnowszej sesji z normą.
${modeBlock}

═══ DANE ═══
Wszystko poniżej podane jest raz; bloki nie powtarzają się nawzajem.

[1] CZAS I KALENDARZ (policzone przez serwer, pewne)
Dziś jest: ${cal.dzis.dzisSlownie}. Godzina: ${cal.dzis.godzinaSlownie}.
${cal.kalendarz}
Notatka o dzisiejszych streamach: ${note}

[2] KONTEKST Z SIECI (od najważniejszego: Polska, sport, Twitch, gry, na końcu gra streamera; tylko potwierdzone fakty, sprawdzone dziś)
${netBlock}

[3] NAJNOWSZA SESJA (wyniki całej sesji; pola „szac_” to szacunki; „glownaGra” = gra z największą liczbą minut; „tytul” = jeden tytuł: najdłużej trwający, a przy trwającej sesji aktualny)
${J(csRest)}
${liveBlock}${przebiegTxt}
[4] HISTORIA SESJI I WYLICZENIA SERWERA (porównania najnowszej sesji z normą, wzorce dni i godzin, średnie ważone)
${historyTable(sessionsHistory)}

Wyliczenia serwera:
${J(derived)}

[5] STATYSTYKI 14 DNI (porównanie ostatnich 14 dni z dniami 15–30 wcześniej; to NIE jest porównanie najnowszej sesji z normą)
${statsBlock(statystyki)}

[6] KANAŁ POZA SESJAMI (followersi, suby, tempo wzrostu; bez nicków)
${J(kanalOut)}

═══ FORMAT ODPOWIEDZI ═══
Zwróć WYŁĄCZNIE gotowy tekst wypowiedzi do przeczytania na głos po polsku: czysty tekst, bez markdownu, wypunktowań, symboli, emoji, nagłówków, JSON-a i adresów stron. Krótkie zdania; liczby, godziny i jednostki zapisane słownie, tak jak się je mówi, bez skrótów i symboli. Mniej liczb, więcej sensu. Zachowaj długość z sekcji TRYB.`;
}

// ═══════════════ GŁOS (TTS) ═══════════════
function pcmToWav(pcm, sampleRate = 24000, channels = 1, bits = 16) {
    const h = Buffer.alloc(44);
    h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVE', 8);
    h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(channels, 22);
    h.writeUInt32LE(sampleRate, 24); h.writeUInt32LE(sampleRate * channels * bits / 8, 28);
    h.writeUInt16LE(channels * bits / 8, 32); h.writeUInt16LE(bits, 34);
    h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
    return Buffer.concat([h, pcm]);
}

function pickAudio(i) {
    if (i?.output_audio?.data) return i.output_audio.data;
    const steps = i?.steps || i?.outputs || [];
    for (const st of [...steps].reverse()) {
        for (const c of [...(st.content || [])].reverse()) if (c?.type === 'audio' && c.data) return c.data;
    }
    return null;
}

// Styl próbujemy z krótkim limitem; gdy zawiedzie, wracamy do wariantu bez stylu i przez 10 min nie marnujemy zapytań na styl.
let styleOffUntil = 0;
const STYLE_TIMEOUT_MS = Number(process.env.TTS_STYLE_TIMEOUT_MS) || 25000;
async function makeAudio(text, models = TTS_MODELS, mode = 'offline') {
    let lastErr = null;
    const style = TTS_STYLES[mode] || TTS_STYLES.offline;
    for (const model of models) {
        if (isBlocked(model)) { lastErr = new Error(`TTS ${model}: 429 RESOURCE_EXHAUSTED — model wstrzymany po przekroczeniu limitu`); continue; }
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
                }), st ? STYLE_TIMEOUT_MS : 45000, `TTS ${model}`);
                const b64 = pickAudio(interaction);
                if (!b64) {
                    lastErr = new Error(`TTS ${model}: brak audio w odpowiedzi`);
                    if (st) { styleOffUntil = Date.now() + 10 * 60e3; console.error(`TTS (${model}) ze stylem bez audio — wyłączam styl na 10 min`); continue; }
                    break;
                }
                const buf = Buffer.from(b64, 'base64');
                const out = buf.slice(0, 4).toString('ascii') === 'RIFF' ? buf : pcmToWav(buf);
                console.log(`TTS OK: ${model}${st ? ` (styl: ${mode})` : ' (bez stylu)'}`);
                return out.toString('base64');
            } catch (e) {
                lastErr = e;
                console.error(`TTS (${model}${st ? ', ze stylem' : ', bez stylu'}) nie zadziałał:`, e.message);
                const b = blockFor(e); if (b) { blocked.set(model, Date.now() + b); break; } // limit/brak modelu → od razu następny model
                if (st) { styleOffUntil = Date.now() + 10 * 60e3; continue; }
                break;
            }
        }
    }
    if (lastErr) throw lastErr;
    return null;
}

// Ten sam tekst + tryb + głos = ten sam plik audio z pamięci (ponowne odsłuchanie analizy nie kosztuje zapytań do TTS)
function makeAudioCached(text, mode) {
    const key = crypto.createHash('sha1').update(`${mode}|${TTS_VOICE}|${text}`).digest('hex');
    return cacheAudio.memo(key, TTL.audio, async () => {
        const a = await makeAudio(text, TTS_MODELS, mode);
        if (!a) throw new Error('Model nie zwrócił audio');
        return a;
    });
}

// ═══════════════ ENDPOINTY ═══════════════
app.get('/', (req, res) => res.send('OK'));
app.get('/health', (req, res) => res.json({ ok: true, textModel: TEXT_MODEL, textModels: TEXT_MODELS, researchModels: RESEARCH_MODELS, ttsModels: TTS_MODELS, think: THINK, wstrzymane: blockedSummary() || null }));
// Ręczne wyczyszczenie blokad modeli (np. po zmianie limitów w Google) — nie wysyła żadnych zapytań do Gemini
app.get('/reset-limits', (req, res) => { blocked.clear(); strikes.clear(); res.json({ ok: true, wyczyszczono: true }); });

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Diagnostyka (uwaga: każde otwarcie to kilka zapytań do Gemini — używaj tylko do testów)
app.get('/diag', async (req, res) => {
    const timed = async (label, fn) => {
        const t0 = Date.now();
        try { await fn(); return { label, ok: true, ms: Date.now() - t0 }; }
        catch (e) { return { label, ok: false, ms: Date.now() - t0, error: e.message }; }
    };
    // /diag?search=1 dodatkowo testuje modele wyszukiwania (zużywa po 1 zapytaniu z limitu wyszukiwania na model)
    const searchList = req.query.models ? list(req.query.models, '') : RESEARCH_MODELS;
    const searchTests = req.query.search ? searchList.map(m => timed('sieć ' + m, () => genOne('Podaj jednym zdaniem dzisiejszą datę w Polsce.', true, 60000, m, 'low'))) : [];
    const out = await Promise.all([
        ...TEXT_MODELS.map(m => timed('tekst ' + m, () => genOne('Odpowiedz jednym słowem: ok', false, 45000, m, 'low'))),
        ...searchTests,
        ...TTS_MODELS.map(m => timed('głos ' + m, () => makeAudio('Cześć, test głosu.', [m])))
    ]);
    res.json(out);
});

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

app.post('/api/tts', async (req, res) => {
    try {
        const text = String(req.body?.text || '').slice(0, 3500);
        const mode = req.body?.mode === 'live' ? 'live' : 'offline';
        if (!text) return res.status(400).json({ success: false, error: 'Brak tekstu' });
        const audio = await makeAudioCached(text, mode);
        res.json({ success: true, audio });
    } catch (error) {
        console.error('Błąd TTS:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.post('/api/analyze', async (req, res) => {
    try {
        const body = normalizeBody(req.body || {});   // nowa paczka w blokach lub stara — dalej zawsze ten sam układ
        const { sessionsHistory = [], currentSession = null, channelName = 'Dolaricious', noData = false, kanal = {}, teraz = {} } = body;
        const PROMPT_V = 'v7';   // zmieniaj po edycji promptu, żeby stare analizy z pamięci podręcznej nie wracały
        const live = body.tryb === 'live' || (body.tryb == null && !!currentSession?.trwaJeszcze);
        const mode = live ? 'live' : 'offline';

        if (noData) {
            const txt = `Hej! Tu analiza kanału ${channelName}. Na razie nie mam zapisanych żadnych zakończonych sesji ani danych kanału, więc nie ma jeszcze czego analizować. Po pierwszej zakończonej transmisji omówię wyniki, a potem szerszy obraz polskiego Twitcha.`;
            return res.json({ success: true, analysis: txt, audioPrompt: txt, searchUsed: false, sources: [], mode });
        }

        const cal = calendarContext();
        // Klucz pamięci podręcznej: te same dane = ta sama analiza (0 zapytań do Gemini)
        const key = live
            ? `A|${PROMPT_V}|live|${currentSession?.start}|${currentSession?.glownaGra || teraz.gra || ''}|${teraz.tytul || currentSession?.tytul || ''}`
            : `A|${PROMPT_V}|off|${currentSession?.start}|${currentSession?.koniec}|${cal.dzis.dataISO}`;
        if (body.odswiez) cacheMain.del(key);

        const result = await cacheMain.memo(key, live ? TTL.analysisLive : TTL.analysisOffline, async () => {
            const tStart = Date.now();
            const warns = [];
            const cs = keepBigSpikes(fixSession(stripNicks(currentSession), warns)), hist = (Array.isArray(sessionsHistory) ? stripNicks(sessionsHistory) : []).map(x => keepBigSpikes(fixSession(x, warns)));
            if (warns.length) console.warn('Niespójności w danych sesji: ' + warns.join(' | '));   // model nie dostaje nicków
            const derived = derive(cs, hist, stripNicks(kanal));
            let ctx = { net: null, sources: [] };
            try { ctx = (await research(body, cal, mode)) || ctx; }   // zwykle z pamięci podręcznej; błąd badania sieci nigdy nie zatrzymuje analizy
            catch (e) { console.error('Badanie sieci — nieoczekiwany błąd, analiza idzie bez niego:', e.message); }
            const searchUsed = !!ctx.net;
            const prompt = buildPrompt({ ...body, kanal: stripNicks(kanal), teraz: stripNicks(teraz), live, currentSession: withClock(tidySession(cs, true)), sessionsHistory: hist.map(x => tidySession(x, false)), statystyki: body.statystyki || null }, cal, derived, ctx);
            // accept: pusta lub zepsuta odpowiedź jednego modelu to tylko jego porażka — serwer od razu pyta następny model z listy
            const r = await genSeq(prompt, false, PER_MODEL_MS, THINK[mode], TEXT_MODELS, TEXT_MODELS.length, tStart + TOTAL_BUDGET_MS, x => !!speechFrom(x?.text));
            const text = speechFrom(r.text);
            if (!text) throw new Error('Pusta odpowiedź modelu');
            // Gdy nie udało się zbadać sieci, nie utrwalamy tej uboższej analizy na długo
            if (!searchUsed) setTimeout(() => cacheMain.del(key), 60e3).unref?.();
            return { text, searchUsed, sources: ctx.sources, ostrzezenia: warns };
        });

        res.json({ success: true, analysis: result.text, audioPrompt: result.text, searchUsed: result.searchUsed, sources: result.sources || [], ostrzezeniaDanych: result.ostrzezenia || [], mode });
    } catch (error) {
        console.error('Błąd podczas analizy:', error);
        if (isQuota(error)) return res.status(429).json({ success: false, error: 'Limit Gemini wyczerpany — spróbuj za chwilę (limit minutowy) lub po ok. 9:00 (limit dzienny). Stan: ai.dev/rate-limit' + (blockedSummary() ? ' · wstrzymane: ' + blockedSummary() : '') });
        res.status(500).json({ success: false, error: error.message });
    }
});

// Ostatnia siatka bezpieczeństwa: nieobsłużony błąd w pojedynczym żądaniu nie może położyć całego serwera.
app.use((err, req, res, next) => {
    console.error('Błąd żądania:', err);
    if (res.headersSent) return next(err);
    res.status(err?.status || 500).json({ success: false, error: err?.message || 'Błąd serwera' });
});
process.on('unhandledRejection', e => console.error('Nieobsłużone odrzucenie obietnicy:', e));
process.on('uncaughtException', e => console.error('Nieobsłużony wyjątek:', e));

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
