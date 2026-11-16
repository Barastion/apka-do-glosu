const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(cors());

// Klucz pobierany ze zmiennej środowiskowej GEMINI_API_KEY (Render → Environment)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Modele można zmienić w Render → Environment bez ruszania kodu (TEXT_MODEL / TTS_MODEL)
const TEXT_MODEL = process.env.TEXT_MODEL || 'gemini-3.8-flash';
const list = (v, def) => String(v || def).split(',').map(x => x.trim()).filter(Boolean);
// GŁOS: główny + zapasowe (każdy model TTS ma osobny limit dzienny). Zmienisz w Render → Environment: TTS_MODEL, TTS_MODEL_FALLBACKS (po przecinku).
const TTS_MODELS = [...new Set([process.env.TTS_MODEL || 'gemini-3.8-flash-tts', ...list(process.env.TTS_MODEL_FALLBACKS, 'gemini-3.8-flash-lite-tts,gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts')])];
const TTS_VOICE = process.env.TTS_VOICE || 'Kore';
// Modele tekstowe są próbowane PO KOLEI (zapasowy tylko gdy główny zawiedzie) — nie równolegle, bo to podwajało zużycie limitu
// Kolejność: główny → zapasowe. Zapasowe z env TEXT_MODEL_FALLBACKS (po przecinku), np. "gemini-3.7-flash,gemini-2.5-flash-lite".
// Każdy model ma OSOBNE limity, więc po wyczerpaniu jednego serwer przechodzi na następny. Model, którego nie ma (404), jest pomijany.
// ANALIZA: najmocniejszy model pierwszy, potem zapasowe (env TEXT_MODEL_FALLBACKS). Flash-Lite mają na Twoim koncie 500 zapytań/dzień.
const TEXT_MODELS = [...new Set([TEXT_MODEL, ...list(process.env.TEXT_MODEL_FALLBACKS || process.env.TEXT_MODEL_FALLBACK, 'gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash,gemini-2.5-flash-lite')])];   // 2.5 = ostatnia warstwa zapasowa
// BADANIE SIECI (wyszukiwanie): proste streszczenie, więc najpierw lekkie modele z dużym limitem dziennym (env RESEARCH_MODELS).
const RESEARCH_MODELS = [...new Set(list(process.env.RESEARCH_MODELS, 'gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.8-flash,gemini-2.5-flash,gemini-2.5-flash-lite'))];   // 2.5 = ostatnia warstwa zapasowa
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
    games: 12 * 3600e3,           // notatki z sieci (kalendarz, gry, rynek) ważne 12 h w ciągu tej samej doby
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
    if (isQuota(e)) {
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
    const chunks = r.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    return { text: r.text, model, via: 'generateContent', sources: chunks.map(c => c.web?.uri).filter(Boolean) };
}

// Modele PO KOLEI: każdy dostaje pełne `ms` (nie więcej niż zostało z budżetu); błąd = od razu następny; pomijamy modele wstrzymane.
// Odrzucenia 429 i 404 przychodzą od razu, więc nie zużywają limitu prób.
// Blokady: limit samego modelu blokuje model wszędzie; ogólny 429 przy WYSZUKIWANIU blokuje model tylko dla wyszukiwania (klucz 'search|model'),
// bo wyczerpany limit wyszukiwarki nie oznacza, że model nie napisze analizy bez niej.
// Gdy WSZYSTKIE modele są wstrzymane krótką („miękką”) blokadą, robimy drugie przejście mimo blokad, zamiast od razu zwracać błąd.
async function genSeq(prompt, search, ms, think, models = TEXT_MODELS, maxAttempts = MAX_ATTEMPTS, deadline = Infinity) {
    const t0 = Date.now(), errs = [];
    let attempts = 0, tried = 0;
    const remaining = m => Math.max(blocked.get(m) || 0, search ? (blocked.get('search|' + m) || 0) : 0) - Date.now();
    const pass = async force => {
        for (const m of models) {
            if (attempts >= maxAttempts) return null;
            const left = deadline - Date.now();
            if (left < 8000) { errs.push('budżet czasu wyczerpany'); return null; }
            const rem = remaining(m);
            if (rem > 0 && (!force || rem > SOFT_BLOCK_MS)) {
                if (!force) errs.push(`${m}: 429 RESOURCE_EXHAUSTED — model wstrzymany po przekroczeniu limitu (jeszcze ok. ${Math.ceil(rem / 60e3)} min)`);
                continue;
            }
            attempts++; tried++;
            try {
                const r = await genOne(prompt, search, Math.min(ms, left), m, think);
                blocked.delete(m); blocked.delete('search|' + m); strikes.delete(m); strikes.delete('search|' + m);
                console.log(`Model OK (${r.model}, ${r.via}${search ? ', z wyszukiwaniem' : ''}, think=${think}): ${Date.now() - t0} ms`);
                return r;
            } catch (e) {
                const msg = String(e?.message || e);
                errs.push(`${m}: ${msg}`); console.error('Model zawiódł:', msg);
                const key = (search && isQuota(e) && !/for model|PerDay|per day|daily/i.test(msg)) ? 'search|' + m : m;
                const b = Math.min(blockForText(e, key), MAX_BLOCK_MS);
                if (b) blocked.set(key, Date.now() + b);
                if (isQuota(e) || /not found|NOT_FOUND|\b404\b/i.test(msg)) attempts--;   // natychmiastowe odrzucenie nie zużywa próby
            }
        }
        return null;
    };
    let r = await pass(false);
    if (!r && tried === 0 && deadline - Date.now() >= 8000) { console.log('Wszystkie modele wstrzymane krótką blokadą — próbuję mimo to'); r = await pass(true); }
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
const MIN_SLOW = { 15: 'piętnaście', 30: 'trzydzieści', 45: 'czterdzieści pięć' };
function godzSlownie(hhmm) {
    const [h, m] = String(hhmm).split(':').map(Number);
    if (!Number.isFinite(h) || !Number.isFinite(m)) return String(hhmm);
    if (h % 24 === 0 && m === 30) return 'pół godziny po północy';
    const base = GODZ_DOP[h % 24];
    return m ? `${base} ${MIN_SLOW[m] || m}` : base;
}
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

function calendarContext(now = new Date()) {
    const p = warsawParts(now), dow = new Date(Date.UTC(+p.year, +p.month - 1, +p.day)).getUTCDay();
    return {
        dzis: {
            data: new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, day: 'numeric', month: 'long', year: 'numeric' }).format(now),
            dzienTygodnia: p.weekday, godzina: `${p.hour}:${p.minute}`, dataISO: `${p.year}-${p.month}-${p.day}`, weekend: dow === 0 || dow === 6
        }
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
// Porównanie „najnowsza sesja vs norma” i gry nadal używają szybszego spadku 0,82. Zmienna środowiskowa PATTERN_DECAY (0–1).
const PATTERN_DECAY = (Number(process.env.PATTERN_DECAY) > 0 && Number(process.env.PATTERN_DECAY) <= 1) ? Number(process.env.PATTERN_DECAY) : 0.95;
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
    // wagi: najnowsza sesja = 1, każda starsza × 0.82
    const items = all.map((s, i) => ({ s, m: sessionMetrics(s), w: Math.pow(0.82, i), wp: Math.pow(PATTERN_DECAY, i) }));   // w: porównania i gry; wp: wzorce (dni, pory, okna)
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
            uwaga: 'Wiadomości i followersi na godzinę streama; średnia widzów i peak bezpośrednio (uwzględnij różnicę długości).',
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
    const slot = s => `${s.godzinaStartu || clock(new Date(s.start))}–${clock(endOf(s))}, trwał ${durFraza(sessMin(s))}${s.glownaGra ? ` (${s.glownaGra})` : ''}`;
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
        const head = `Stream trwa teraz. Start: ${current.dzienTygodnia}, godz. ${current.godzinaStartu} (${temu(startD.getTime(), now)}).`;
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
    if (todayList.length === 1) return `Dziś był już stream: start ${current.godzinaStartu}, koniec ok. ${clock(endD)}, trwał ${durFraza(sessMin(current))}, skończył się ${temu(endD.getTime(), now)}.`;
    return `Dziś nie było jeszcze streama. Ostatni skończył się ${temu(endD.getTime(), now)} (${current.dzienTygodnia}, start ${current.godzinaStartu}, koniec ok. ${clock(endD)}).`;
}

// ═══════════════ BADANIE SIECI (z pamięcią podręczną) ═══════════════
// Zamiast wyszukiwać w sieci przy KAŻDYM kliknięciu, robimy to osobno i zapamiętujemy:
//  • kalendarz/święta/wydarzenia — raz na dzień (model sam sprawdza w sieci, żadnych wgranych dat, działa w każdym roku),
//  • rynek gier — raz na ~12 godzin dla danego zestawu gier.
// Analiza końcowa nie używa już wyszukiwarki, więc jest szybka i tania.
// JEDNO zapytanie z wyszukiwaniem (kalendarz + gry + rynek) zamiast dwóch — darmowy plan ma tylko ok. 20 zapytań dziennie na model.
function researchPrompt(cal, games) {
    return `Dziś jest ${cal.dzis.dzienTygodnia}, ${cal.dzis.data}, godz. ${cal.dzis.godzina} (czas polski). Sprawdź w sieci i podaj WYŁĄCZNIE fakty potwierdzone dla DZISIAJ i najbliższych 7 dni w Polsce.
CZĘŚĆ 1 — KALENDARZ: święta państwowe i kościelne, dni wolne, długi weekend lub „most”; wakacje, ferie (zimowe z podziałem na województwa), okres przedświąteczny, matury i egzaminy, pora roku; wydarzenia odciągające Polaków od Twitcha lub ściągające na niego: mecze reprezentacji Polski, duże gale freak-fightowe, finały e-sportowe (np. IEM Katowice), ważne premiery i największe wydarzenia na Twitchu. Zacznij od jednego zdania: czy dziś to dzień roboczy, weekend czy dzień wolny.
${games.length ? `CZĘŚĆ 2 — GRY streamowane ostatnio na polskim kanale (od najświeższej): ${games.join('; ')}. Dla każdej: oglądalność w Polsce i miejsce na tle czołówki polskiego Twitcha; stosunek widzów do streamujących i kiedy bywa najkorzystniejszy (tylko informacja); patche, aktualizacje, premiery, dropy i eventy DZIŚ, w przyszłości lub z ostatnich 7 dni (jeśli realnie wpływają na Twitcha). Dodatkowo: jaka inna duża gra lub wydarzenie na Twitchu ściąga teraz widzów w Polsce.` : 'CZĘŚĆ 2 — pomiń gry (brak danych); podaj tylko, jaka duża gra lub wydarzenie na Twitchu ściąga teraz widzów w Polsce.'}
Użyj maksymalnie 4 wyszukiwań. Odpowiedz krótkimi punktami (maks. 260 słów) z podziałem na „KALENDARZ” i „GRY”. Nie zmyślaj liczb; czego nie da się sprawdzić, pomiń; jeśli w kalendarzu nic nie potwierdzisz, pomiń „KALENDARZ”. Bez nazw firm, portali i serwisów (wyjątek: gry, studia, wydawcy, Twitch) i bez linków.`;
}
function pickGames(body) {
    const seen = [], add = g => { if (g && typeof g === 'string' && !seen.some(x => x.toLowerCase() === g.toLowerCase())) seen.push(g.trim()); };
    const cur = body.currentSession;
    add(body.teraz?.gra); add(cur?.glownaGra); (cur?.gry || []).slice(0, 2).forEach(x => add(x.nazwa));
    (body.sessionsHistory || []).slice(0, 4).forEach(s => add(s.glownaGra));
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
    const date = cal.dzis.dataISO;
    const games = pickGames(body);
    const gk = games.map(x => x.toLowerCase()).sort().join('+');
    const key = `net|${date}|${gk}`;   // ważne 12 h w ciągu tej samej doby, osobno dla danego zestawu gier
    try {
        const r = await cacheMain.memo(key, TTL.games, async () => {
            const x = await genSeq(researchPrompt(cal, games), true, RESEARCH_MS, 'low', RESEARCH_MODELS, 4, Date.now() + RESEARCH_BUDGET_MS);
            lastNet = { date, gk, text: x.text?.trim() || '', sources: x.sources || [], at: Date.now() };
            return x;
        });
        return { net: r.text?.trim() || null, sources: r.sources || [] };
    } catch (e) {
        console.error('Badanie sieci nieudane:', e.message);
        if (lastNet && lastNet.text && lastNet.date === date && Date.now() - lastNet.at < 12 * 3600e3) {
            const h = Math.max(1, Math.round((Date.now() - lastNet.at) / 36e5));
            console.log('Używam ostatnich zapamiętanych notatek z sieci (sprzed ' + h + ' h)');
            return { net: lastNet.text + `\n(Uwaga: te notatki z sieci pochodzą sprzed ok. ${h} godz., mogły się zmienić${lastNet.gk !== gk ? ' i część o grach może dotyczyć innych gier' : ''}.)`, sources: lastNet.sources };
        }
        return { net: null, sources: [] };
    }
}

// ═══════════════ HISTORIA JAKO ZWARTA TABELA (zamiast JSON-a: ok. 3–4× mniej tokenów, a każda sesja ma dzień, godziny i grę) ═══════════════
const DOW3 = { poniedziałek: 'pon', wtorek: 'wt', środa: 'śro', czwartek: 'czw', piątek: 'pią', sobota: 'sob', niedziela: 'nie' };
const cellTxt = v => (v === undefined || v === null || v === '') ? '-' : String(v).replace(/\|/g, '/');
function sessionRow(s) {
    const st = new Date(s.start), en = s.koniec ? new Date(s.koniec) : new Date(st.getTime() + (s.czasTrwaniaMin || 0) * 6e4);
    let game = s.glownaGra || '-';
    if (Array.isArray(s.gry) && s.gry.length > 1) {   // druga gra tylko gdy zajęła sporą część streama
        const extra = s.gry.slice(1).filter(g => g.minuty >= Math.max(20, (s.czasTrwaniaMin || 0) * 0.2)).map(g => `${g.nazwa} ${g.minuty} min`);
        if (extra.length) game += ` (+${extra.join(', ')})`;
    }
    const tyt = Array.isArray(s.tytuly) && s.tytuly.length ? [...new Set(s.tytuly)].slice(-2).map(t => String(t).slice(0, 50)).join(' / ') : '';
    return [warsawYmd(st), DOW3[s.dzienTygodnia] || s.dzienTygodnia, `${s.godzinaStartu || clock(st)}–${s.godzinaKonca || clock(en)}`, s.czasTrwaniaMin, game,
        s.sredniaWidzow, s.peakWidzow, s.wiadomosciNaCzacie, s.unikalniCzatujacy, s.nowiFollowersi, s.noweSuby ?? s.noweSubskrypcje,
        s.szac_unikalniWidzowie, s.szac_sredniCzasOgladaniaMin, tyt, Array.isArray(s.skokiWidzow) ? s.skokiWidzow.length : undefined].map(cellTxt).join('|');
}
const historyTable = h => (h || []).length
    ? `HISTORIA SESJI (najnowsze pierwsze, jedna sesja = jedna linia, "-" = brak danych). Kolumny: data|dzień|od–do|czas w min|gra|średnia widzów|peak|wiadomości na czacie|piszący|nowi followersi|nowe suby|szac. unikalni widzowie|szac. średni czas oglądania w min|tytuł|liczba skoków widowni\n${h.map(sessionRow).join('\n')}`
    : 'HISTORIA SESJI: brak.';

// ═══════════════ PROMPT ANALIZY ═══════════════
function buildPrompt(body, cal, derived, ctx) {
    const { channelName = 'Dolaricious', sessionsHistory = [], currentSession = null, live = false, kanal = {}, teraz = {}, wariantCTA = 'obserwowanie' } = body;
    const note = lastSessionNote(currentSession, cal, live, sessionsHistory);
    const ctaTxt = wariantCTA === 'obserwowanie_i_suby'
        ? 'subtelnie zachęć widzów do zaobserwowania kanału ORAZ do zostawienia suba (jedno–dwa miękkie zdania)'
        : 'subtelnie zachęć widzów wyłącznie do zaobserwowania kanału (jedno miękkie zdanie, bez wzmianki o subskrypcji)';

    const modeBlock = live ? `
TRYB: STREAM TRWA (na żywo). Widzowie prawdopodobnie też Cię słyszą.
- Skup się na bieżącej sesji na tle historii. Obowiązkowo wspomnij: aktualną liczbę widzów, średnią sesji, wiadomości na czacie i liczbę piszących, nowe followy i suby w tej sesji (jeśli są), o której zaczął się stream, dzień tygodnia, grę i tytuł — ale jak w audycji, a nie jak z tabeli: widzów i średnią powiedz razem w jednym zdaniu, wiadomości i piszących w drugim (zaokrąglone), followy i suby przy okazji, tytuł krótkim zwrotem. W całej wypowiedzi użyj najwyżej ok. sześciu liczb, resztę przekaż słowami („czat żyje”, „widownia rośnie”). Jeśli dziś był już wcześniejszy stream, wspomnij o nim jednym zdaniem (z czasem trwania wg gotowej frazy).
- Z pola "aktualnie" (słupki widzów i czatu, trendy) powiedz, czy widownia i czat rosną, maleją czy są stabilne.
- Z KONTEKSTU Z SIECI podaj, ile osób w Polsce ogląda tę grę i ilu ją streamuje (jeśli jest w kontekście), oraz największe wydarzenia dziś.
- Styl: jak prowadząca audycji radiowej — ciepły, żywy, lekko komentatorski, z małą historią (dzieje się to i to, bo…) zamiast listy wyników. Krótkie, rytmiczne zdania, zwroty do słuchaczy dozwolone. Liczby jako akcenty, nie jako szkielet.
- Zakończ lekkim CTA: ${ctaTxt}. Nienachalnie, bez błagania, tym samym tonem i w tym samym stylu co cała wypowiedź; łagodność oddaj doborem słów.
- Długość: ok. 150–180 słów (ok. minuty mówienia).` : `
TRYB: STREAM ZAKOŃCZONY / OFFLINE.
- Ok. 50% wypowiedzi: dzisiejsze streamy łącznie (jeśli było ich kilka; czas trwania i sumę podaj wg gotowych fraz) i najnowsza zakończona sesja — jak poszła, porównanie z historią (z uwzględnieniem czasu trwania), powody odchyleń.
- Gdy danych o sesjach mało, skup się na danych kanału: followersi, tempo przyrostu w 30 dni, suby, daty ostatnich followów i subów.
- Pozostałe 50%: wzorce (najlepsze dni, pory i godziny wg punktu A, gry, przyczyny), rynek polski, kalendarz, patche/premiery/wydarzenia, stosunek widzów do streamujących dla gier, w które gra.
- Styl: analityczny, spokojny, rzeczowy, ale ludzki — jak rzeczowa rozmowa z doświadczonym analitykiem, nie jak odczyt raportu. Każdy akapit to jeden wątek: wniosek, potem jedna–dwie liczby i przyczyna. W całej wypowiedzi użyj najwyżej ok. dwunastu liczb; reszta słowami. Bez CTA do widzów.
- Długość: ok. 250–300 słów (do 2 minut mówienia).`;

    // Przebieg bieżącej sesji jako zwarta tabela (JSON z powtarzanymi nazwami pól kosztował ok. 2× więcej tokenów)
    // Do modelu nie idą: surowy profil godzinowy (godziny liczy kod), ISO-koniec (jest godzinaKonca) i lista gier, gdy była tylko jedna (to samo co glownaGra)
    const { przebieg: pbRaw, godzinyProfil: _gp, koniec: _ko, ...csRest } = currentSession || {};
    if (Array.isArray(csRest.gry) && csRest.gry.length < 2) delete csRest.gry;
    // Domyślnie CAŁY przebieg (do 48 punktów) — to dane o tym, jak szedł najnowszy stream. Zmienna środowiskowa PRZEBIEG_LINII (np. 24) łączy sąsiednie punkty, jeśli kiedyś trzeba zaoszczędzić tokeny.
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
        ? `PRZEBIEG NAJNOWSZEJ SESJI (jedna linia = jeden odcinek; kolumny: godzina|minuta od startu|śr. widzów|wiadomości w odcinku|skumulowani piszący):\n${thinPb(pbRaw).map(x => [x.godzina, x.min, x.v, x.msg, x.cz].map(cellTxt).join('|')).join('\n')}`
        : '';
    const netBlock = ctx.net
        ? `KALENDARZ, WYDARZENIA, GRY I RYNEK TWITCHA (sprawdzone w sieci na dziś):\n${ctx.net}`
        : 'Danych z sieci (kalendarz, święta, gry, patche, wydarzenia, stosunek widzów do streamujących)  brak — nie wspominaj o tym, pomiń te wątki i niczego nie zmyślaj.';

    return `Jesteś analityczką kanału Twitch streamera ${channelName} (polski rynek streamingowy). Mówisz kobiecym głosem, więc o sobie zawsze w formach żeńskich („przeanalizowałam”, „nie udało mi się”), nigdy w męskich. Masz dane z widżetu, wyliczenia z kodu oraz zweryfikowany w sieci kontekst. Nie wyszukujesz niczego sama — korzystaj tylko z poniższych danych.

═══ ZASADY BEZWGLĘDNE ═══
1. Nigdy nie czytaj ani nie wymieniaj nicków i nazw użytkowników (poza nazwą ${channelName}). Nigdy nie mów o zarobkach, kwotach ani pieniądzach z subskrypcji, bitsów itp. Liczby followersów i subów wolno podawać.
2. Zakaz kryptoreklamy: nie wymieniaj nazw firm, portali, serwisów, narzędzi, sklepów, marek ani produktów. Wyjątki: tytuły gier, studia i wydawcy gier oraz sam Twitch. Źródła opisuj ogólnie (np. „publiczne statystyki Twitcha”).
3. Nigdy nie doradzaj zmiany gry ani przeniesienia streamowania na inną porę lub dzień. WOLNO i warto: zauważać zmiany aktywności w trakcie jego streama (np. wzrost około 12:00, spadek około 14:00) z prawdopodobnymi przyczynami (koniec pracy, dojazd, obiad, lekcje, początek weekendu), informować o stosunku streamujących do widzów i o wydarzeniach — jako obserwacje („dane pokazują…”), bez poleceń zmian.
4. Prawda, bez słodzenia, ale i bez pesymizmu: szukaj plusów, wspominaj minusy. Nie zmyślaj liczb. Odróżniaj twarde dane z historii, szacunki (pola „szac_”) i hipotezy. Czego nie da się sprawdzić — powiedz wprost. Grupy z polem "uwaga" (pojedyncza sesja, tylko dwie sesje) to ciekawostka, nie wzorzec.
5. Świeżość i początek: dziś jest ${cal.dzis.dzienTygodnia}, ${cal.dzis.data}, godz. ${cal.dzis.godzina} (czas polski). Zacznij od przedstawienia się jako analityczka kanału ${channelName} i powiedz, jaki jest dziś dzień, data i godzina. Następnie: ${note} Wspominaj tylko wydarzenia z kontekstu z sieci (dzisiejsze, przyszłe lub z ostatnich 7 dni); niepotwierdzonego nie wspominaj.
6. Dopasuj analizę do polskich widzów i polskiego Twitcha.
7. Czas, który upłynął od streama, podawaj wyłącznie gotową frazą z danych (np. „25 minut temu”, „3 godziny i 45 minut temu”). Nie przeliczaj go sama i nie zaokrąglaj do godzin. To samo dotyczy CZASU TRWANIA streamów i ich sumy z jednego dnia: podawaj wyłącznie gotowe frazy z danych („czasFraza”, „Łączny czas streamowania dziś”, „trwał …”). Nigdy nie dodawaj ani nie odejmuj godzin i minut samodzielnie, a gdy gotowej frazy brak — nie podawaj sumy.
8. Najnowszy stream porównuj do historii jako grupy, nigdy do pojedynczych dawnych sesji ani dni (także gdy "dniZWieluStreamami" dotyczy innego dnia). Wyjątek: inne streamy z TEGO SAMEGO DNIA co najnowszy — oceń dzień łącznie, potem sesje osobno wartościami na godzinę; różnicę tłumacz porą i przerwą, nie traktuj jej jako samodzielnego wniosku.
9. Pora i konkurencja: piątkowy wieczór, sobota, niedziela i święta to nie tylko więcej oglądających, ale też więcej streamujących naraz, bo jedno idzie w parze z drugim. Zakładaj to zawsze, także gdy brakuje danych z sieci. Uwzględniaj to w szukaniu przyczyn jako prawdopodobną hipotezę, obok innych możliwych przyczyn.
10. Mów o liczbach z sensem, dla słuchacza, który słyszy tekst raz i nie może wrócić do poprzedniego zdania. Najpierw wniosek własnymi słowami, potem liczba, która go potwierdza („widownia była wyraźnie większa niż zwykle, średnio o około dwadzieścia procent”). Każda liczba ma od razu mieć znaczenie: powiedz, co oznacza, a nie tylko ją podaj. Liczby i procenty są mile widziane, ale oszczędnie i tam, gdzie coś pokazują: nie w każdym zdaniu, najwyżej jedna–dwie w zdaniu i nigdy kilka procentów pod rząd. Zaokrąglaj jak w rozmowie („prawie czterdzieści osób”, „co trzeci”); dokładnie podawaj tylko czas trwania i godziny. Nie czytaj nazw pól, indeksów ani współczynników — zamień je na zwykłe słowa. Z danych wybierz to, co najważniejsze, i pomiń resztę.
11. GODZINY: zawsze w systemie 24-godzinnym, słownie, tak jak w danych (17:30 = „siedemnasta trzydzieści”, nigdy „piąta trzydzieści” ani „piąta po południu”; bez „rano”, „po południu”, „w nocy” przy godzinach). Gdy w danych jest gotowe pole „okresSlownie”, „oknoSlownie” lub „godzinaSlownie”, przepisz je bez zmian. Nie zmieniaj, nie przesuwaj i nie zgaduj godzin; podawaj tylko te, które są w danych.
12. OCENA: na koniec analizy wystaw subiektywną ocenę ogólną najnowszego streama w skali od 1 do 10 (liczba całkowita; 10 = stream bardzo dobry, 1 = bardzo słaby) na tle wszystkich sesji historycznych, na podstawie całości danych, bez liczenia. Podaj ją słownie i krótko uzasadnij.

═══ CO ANALIZOWAĆ ═══
A. Najlepsze momenty (pole "najlepszeMomenty", nie licz w pamięci): wskaż maks. 3 najlepsze dni tygodnia, maks. 2 najlepsze pory dnia (rano, południe, popołudnie, wieczór, noc) oraz godziny warte uwagi ("mocneOkna" = najmocniejsze okna półgodzinne, "mocnePrzedzialy" = ciągłe przedziały godzin z widownią wyraźnie powyżej średniej streama, także w środku sesji; podawaj przedziały gotową frazą z „okresSlownie”, np. „od dwudziestej do dwudziestej drugiej trzydzieści”). Nie wymieniaj więcej. Poprzyj to jedną lub dwiema najważniejszymi liczbami z danych, zaokrąglonymi i podanymi po ludzku (nie czytaj indeksów); bez danych niczego nie wskazuj. Tylko wspomnij, że to mocne dni i godziny według danych — bez sugerowania, kiedy warto być online. Dni i godziny to dwa OSOBNE wnioski: podaj je osobno („mocne dni”, „mocne godziny”) i nigdy nie łącz ich w „wtorek o dwudziestej” — dane nie sprawdzają tej kombinacji. Gdy widownia jest mała, a czatIdx, followyIdx, wiadomosciNaWidzaNaGodzine lub followyNa100GodzinOgladania wysokie, powiedz, że to mała, ale zaangażowana widownia. Godziny liczone są z wielu zapisanych sesji ("sesjiZPrzebiegiem") — możesz powiedzieć, z ilu. Dołącz "followyMomenty", jeśli pasuje do tych godzin. "odStartuH" mówi, ile godzin po starcie streama wypadała dana godzina. Przy małej próbie ("wiarygodnosc") mów ostrożnie („na razie wygląda na…”), ale nadal wskaż najlepsze. Szukaj PRZYCZYN (początek weekendu, święto, wakacje/ferie, pora roku, wydarzenia z kontekstu, ruch i konkurencja wg zasady 9).
A2. Wzorce WEWNĄTRZ streama: w tabeli PRZEBIEG NAJNOWSZEJ SESJI (każda linia ma godzinę) i w słupkach z "aktualnie" znajdź wzrosty i spadki widowni i czatu, podaj godzinę i prawdopodobną przyczynę. Pierwszą godzinę streama („rozped”) traktuj jako rozpęd: niższa widownia na początku jest normalna i nie jest słabością, ale dane z niej są cenne.
B. Gry: nowsze mają większą wagę (policzona). Wykorzystaj kontekst o oglądalności, stosunku widzów do streamujących, patchach i eventach.
C. Tytuł i gra a wyniki: przy „DROP”/„drops” w tytule sprawdź, czy wzrost widowni przełożył się na piszących. Widzów więcej, a czat bez zmian = widzowie głównie „po nagrody”; rosną oba = stream faktycznie trzyma.
D. Korelacje i rajdy: więcej widzów + więcej wiadomości = dobry znak; mało widzów, ale obecnych długo = stabilna publiczność; mało widzów i pisze ta sama grupa (wysoki udział czatujących) = własne grono. Wskaż, która sytuacja wygląda na prawdziwą. "skokiWidzow" = nagły wzrost z szybkim odpływem: powiedz, że to prawdopodobnie rajd lub hosting (hipoteza) i jak wygląda widownia „bazowa” bez skoku. Bez nicków.
F. Czas trwania i przyczyny: nie porównuj sesji 1:1 — używaj wartości na godzinę z "porownanieNajnowszejSesji" i wspominaj o różnicy długości. Zawsze szukaj PRZYCZYN odchyleń, nie samego „o 5% więcej niż zwykle”.
G0. Kiedy wpadają followy (pole "followyMomenty", ostatnie 30 dni, czas polski): "naGodzineStreamu" to followy w przeliczeniu na godzinę streamowania (najlepsze dni, pory, godziny), "najwiecejOgolem" to kiedy wpada ich najwięcej w ogóle, także poza streamem. Wskaż najlepsze (maks. 3 dni, 2 pory, 3 godziny) z jedną–dwiema liczbami i prawdopodobnymi przyczynami. Przy małej próbie (pole "probka", poniżej 15 followów) powiedz, że to za mało danych na wnioski; nie przesadzaj ze znaczeniem pojedynczych liczb. Dane obejmują dokładnie ostatnie 30 dni — nigdy nie mów o 60 dniach.
G. Tempo rozwoju: followersi w 30 dni i wzrost procentowy, saldo po unfollowach, suby (nowe i wygasłe), data ostatniego followa i suba. Nazwij dobry okres, jeśli rosną. Zaznacz, że część danych subów jest przybliżona.
${modeBlock}

═══ DANE ═══
KONTEKST Z SIECI:
${netBlock}

CZAS I STAN TERAZ:
${J(teraz)}

KANAŁ POZA SESJAMI:
${J(derived && derived.followyMomenty ? { ...kanal, followyKiedy: undefined } : kanal)}

NAJNOWSZA SESJA (podsumowanie):
${J(csRest)}

${przebiegTxt}

${historyTable(sessionsHistory)}

WYLICZENIA SERWERA (średnie ważone, normalizacja na godzinę, porównania %):
${J(derived)}

═══ FORMAT ODPOWIEDZI ═══
Zwróć WYŁĄCZNIE gotowy tekst wypowiedzi do przeczytania na głos po polsku — czysty tekst, bez markdownu, wypunktowań, symboli, emoji, nagłówków, JSON-a i adresów stron. Krótkie zdania; liczby, godziny i jednostki zapisane słownie, tak jak się je mówi („dwadzieścia procent”, „osiemnasta trzydzieści”), bez skrótów i symboli. Pamiętaj o zasadzie 10: mniej liczb, więcej sensu. Zachowaj długość z sekcji TRYB.`;
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
app.get('/health', (req, res) => res.json({ ok: true, textModel: TEXT_MODEL, ttsModels: TTS_MODELS, think: THINK, wstrzymane: blockedSummary() || null }));
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
    const out = await Promise.all([
        ...TEXT_MODELS.map(m => timed('tekst ' + m, () => genOne('Odpowiedz jednym słowem: ok', false, 45000, m, 'low'))),
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
        const body = req.body || {};
        const { sessionsHistory = [], currentSession = null, channelName = 'Dolaricious', noData = false, kanal = {}, teraz = {} } = body;
        const PROMPT_V = 'v3';   // zmieniaj po edycji promptu, żeby stare analizy z pamięci podręcznej nie wracały
        const live = body.tryb === 'live' || (body.tryb == null && !!currentSession?.trwaJeszcze);
        const mode = live ? 'live' : 'offline';

        if (noData) {
            const txt = `Cześć! Tu analityczka kanału ${channelName}. Na razie nie mam zapisanych żadnych zakończonych sesji ani danych kanału, więc nie ma jeszcze czego analizować. Po pierwszej zakończonej transmisji omówię wyniki, a potem szerszy obraz polskiego Twitcha.`;
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
            const ctx = await research(body, cal, mode);           // zwykle z pamięci podręcznej
            const searchUsed = !!ctx.net;
            const prompt = buildPrompt({ ...body, kanal: stripNicks(kanal), teraz: stripNicks(teraz), live, currentSession: withClock(cs), sessionsHistory: hist }, cal, derived, ctx);
            const r = await genSeq(prompt, false, PER_MODEL_MS, THINK[mode], TEXT_MODELS, TEXT_MODELS.length, tStart + TOTAL_BUDGET_MS);
            const parsed = String(r.text).trim().startsWith('{') ? parseJson(r.text) : null;
            const text = cleanSpeech(parsed?.audioPrompt || parsed?.textResponse || r.text);
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

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
