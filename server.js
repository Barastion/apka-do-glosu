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
const TTS_STYLES = {
    live: 'warm, energetic female radio presenter, lightly sports-commentator style, engaging and upbeat, lively pace; the final call-to-action sentences are soft, friendly and unobtrusive',
    offline: 'calm, analytical, professional female voice, clear and measured pace, natural and friendly'
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

// ═══════════════ DATA I CZAS ═══════════════
const TZ = 'Europe/Warsaw';
function warsawParts(d) {
    const f = new Intl.DateTimeFormat('pl-PL', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'long', hour: '2-digit', minute: '2-digit', hour12: false });
    return Object.fromEntries(f.formatToParts(d).map(x => [x.type, x.value]));
}
const warsawYmd = d => { const p = warsawParts(d); return `${p.year}-${p.month}-${p.day}`; };
const dayDiff = (a, b) => Math.round((Date.parse(warsawYmd(a) + 'T00:00:00Z') - Date.parse(warsawYmd(b) + 'T00:00:00Z')) / 864e5);
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

function withClock(s) {
    if (!s || !Array.isArray(s.przebieg) || !s.start) return s;
    const t0 = Date.parse(s.start);
    return { ...s, przebieg: s.przebieg.map(x => ({ godzina: clock(new Date(t0 + (x.min || 0) * 6e4)), ...x })) };
}

// ═══════════════ WYLICZENIA STATYSTYCZNE (w kodzie, model tylko interpretuje) ═══════════════
const round = (n, d = 0) => Number.isFinite(n) ? Math.round(n * 10 ** d) / 10 ** d : null;
const pct = (a, b) => (Number.isFinite(a) && Number.isFinite(b) && b > 0) ? Math.round((a - b) / b * 100) : null;
const band = hh => hh >= 6 && hh < 12 ? 'rano (6–11)' : hh >= 12 && hh < 17 ? 'popołudnie (12–16)' : hh >= 17 && hh < 22 ? 'wieczór (17–21)' : 'noc (22–5)';

function sessionMetrics(s) {
    const h = Math.max(0.05, (s.czasTrwaniaMin || 0) / 60);
    const subs = s.noweSuby ?? s.noweSubskrypcje ?? 0;
    return {
        h, avg: s.sredniaWidzow || 0, peak: s.peakWidzow || 0,
        msgH: (s.wiadomosciNaCzacie || 0) / h, folH: (s.nowiFollowersi || 0) / h, subH: subs / h,
        chatShare: s.szac_unikalniWidzowie > 0 ? (s.unikalniCzatujacy || 0) / s.szac_unikalniWidzowie : null,
        msgPerViewerHour: (s.sredniaWidzow > 0) ? (s.wiadomosciNaCzacie || 0) / (s.sredniaWidzow * h) : null
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
        noweSubyNaGodzine_wazone: round(wavg(arr, 'subH'), 2)
    })).sort((a, b) => (b.sredniaWidzow_wazona || 0) - (a.sredniaWidzow_wazona || 0)).slice(0, top);
}

function derive(current, history) {
    const all = [...(current ? [current] : []), ...(history || [])].filter(s => s && s.czasTrwaniaMin > 0 && s.start);
    // wagi: najnowsza sesja = 1, każda starsza × 0.82
    const items = all.map((s, i) => ({ s, m: sessionMetrics(s), w: Math.pow(0.82, i) }));
    const histItems = items.filter(it => it.s !== current);
    const hourOf = s => parseInt(String(s.godzinaStartu || '').slice(0, 2), 10);
    const out = { liczbaSesjiWHistorii: histItems.length };
    out.wedlugGier = groupBy(items, it => it.s.glownaGra);
    out.wedlugDniTygodnia = groupBy(items, it => it.s.dzienTygodnia);
    out.wedlugPorStartu = groupBy(items, it => Number.isFinite(hourOf(it.s)) ? band(hourOf(it.s)) : null);
    out.wedlugGodzinyStartu = groupBy(items, it => Number.isFinite(hourOf(it.s)) ? String(hourOf(it.s)).padStart(2, '0') + ':00' : null);
    if (current && histItems.length) {
        const cm = sessionMetrics(current);
        const base = k => wavg(histItems, k);
        const mk = arr => arr.length ? {
            sredniaWidzow: round(wavg(arr, 'avg'), 1), peak: round(wavg(arr, 'peak'), 1), wiadomosciNaGodzine: round(wavg(arr, 'msgH'), 1),
            nowiFollowersiNaGodzine: round(wavg(arr, 'folH'), 2), liczbaSesjiWBazie: arr.length
        } : null;
        out.porownanieNajnowszejSesji = {
            uwaga: 'Wiadomości i followersi porównane na godzinę streama; średnia widzów i peak bezpośrednio — pamiętaj o różnicy długości sesji.',
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
Użyj maksymalnie 4 wyszukiwań. Odpowiedz krótkimi punktami (maks. 260 słów) z podziałem na „KALENDARZ” i „GRY”. Nie zmyślaj liczb; czego nie da się sprawdzić, napisz wprost; jeśli w kalendarzu nic nie potwierdzisz, napisz „Brak potwierdzonych wydarzeń”. Bez nazw firm, portali i serwisów (wyjątek: gry, studia, wydawcy, Twitch) i bez linków.`;
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

// ═══════════════ PROMPT ANALIZY ═══════════════
function buildPrompt(body, cal, derived, ctx) {
    const { channelName = 'Dolaricious', sessionsHistory = [], currentSession = null, live = false, kanal = {}, teraz = {}, wariantCTA = 'obserwowanie' } = body;
    const note = lastSessionNote(currentSession, cal, live);
    const ctaTxt = wariantCTA === 'obserwowanie_i_suby'
        ? 'subtelnie zachęć widzów do zaobserwowania kanału ORAZ do zostawienia suba (jedno–dwa miękkie zdania)'
        : 'subtelnie zachęć widzów wyłącznie do zaobserwowania kanału (jedno miękkie zdanie, bez wzmianki o subskrypcji)';

    const modeBlock = live ? `
TRYB: STREAM TRWA (na żywo). Widzowie prawdopodobnie też Cię słyszą.
- Skup się na bieżącej sesji na tle historii. Obowiązkowo wspomnij: aktualną liczbę widzów, średnią sesji, wiadomości na czacie i liczbę piszących, nowe followy i suby w tej sesji (jeśli są), o której zaczął się stream, dzień tygodnia, grę i tytuł.
- Z pola "aktualnie" (słupki widzów i czatu, trendy) powiedz, czy widownia i czat rosną, maleją czy są stabilne.
- Z KONTEKSTU Z SIECI podaj, ile osób w Polsce ogląda tę grę i ilu ją streamuje (jeśli jest w kontekście), oraz największe wydarzenia dziś.
- Styl: radiowy, lekko komentatorski, angażujący, kobiecy, ale rzeczowy. Krótkie, rytmiczne zdania.
- Zakończ lekkim CTA: ${ctaTxt}. Nienachalnie, bez błagania.
- Długość: ok. 150–180 słów (ok. minuty mówienia).` : `
TRYB: STREAM ZAKOŃCZONY / OFFLINE.
- Ok. 50% wypowiedzi: najnowsza zakończona sesja — jak poszła, porównanie z historią (z uwzględnieniem czasu trwania), powody odchyleń.
- Gdy danych o sesjach mało, skup się na danych kanału: followersi, tempo przyrostu w 30 dni, suby, daty ostatnich followów i subów.
- Pozostałe 50%: wzorce (najlepsze godziny i dni, gry, przyczyny), rynek polski, kalendarz, patche/premiery/wydarzenia, stosunek widzów do streamujących dla gier, w które gra.
- Styl: analityczny, spokojny, rzeczowy. Bez CTA do widzów.
- Długość: ok. 280–320 słów (do 2 minut mówienia).`;

    const netBlock = ctx.net
        ? `KALENDARZ, WYDARZENIA, GRY I RYNEK TWITCHA (sprawdzone w sieci na dziś):\n${ctx.net}`
        : 'Wyszukiwarka była niedostępna: bieżących danych z sieci (kalendarz, święta, gry, patche, wydarzenia, stosunek widzów do streamujących) NIE udało się sprawdzić — powiedz to krótko i niczego nie zmyślaj.';

    return `Jesteś analityczką kanału Twitch streamera ${channelName} (polski rynek streamingowy). Mówisz kobiecym głosem, więc zawsze używasz żeńskich form gramatycznych. Masz dane z widżetu, wyliczenia z kodu oraz zweryfikowany w sieci kontekst. Nie wyszukujesz niczego sam — korzystaj tylko z poniższych danych.

═══ ZASADY BEZWZGLĘDNE ═══
1. Nigdy nie czytaj ani nie wymieniaj nicków i nazw użytkowników (poza nazwą ${channelName}). Nigdy nie mów o zarobkach, kwotach ani pieniądzach z subskrypcji, bitsów itp. Liczby followersów i subów wolno podawać.
2. Zakaz kryptoreklamy: nie wymieniaj nazw firm, portali, serwisów, narzędzi, sklepów, marek ani produktów. Wyjątki: tytuły gier, studia i wydawcy gier oraz sam Twitch. Źródła opisuj ogólnie (np. „publiczne statystyki Twitcha”).
3. Nigdy nie doradzaj zmiany gry ani przeniesienia streamowania na inną porę lub dzień. WOLNO i warto: zauważać zmiany aktywności w trakcie jego streama (np. wzrost około 12:00, spadek około 14:00) z prawdopodobnymi przyczynami (koniec pracy, dojazd, obiad, lekcje, początek weekendu), informować o stosunku streamujących do widzów i o wydarzeniach — jako obserwacje („dane pokazują…”), bez poleceń zmian.
4. Prawda, bez słodzenia, ale i bez pesymizmu: szukaj plusów, wspominaj minusy. Nie zmyślaj liczb. Odróżniaj twarde dane z historii, szacunki (pola „szac_”) i hipotezy. Czego nie da się sprawdzić — powiedz wprost.
5. Świeżość: dziś jest ${cal.dzis.dzienTygodnia}, ${cal.dzis.data}, godz. ${cal.dzis.godzina} (czas polski). Wspominaj tylko wydarzenia z kontekstu z sieci (dzisiejsze, przyszłe lub z ostatnich 7 dni). Niepotwierdzonego nie wspominaj.
6. Dopasuj analizę do polskich widzów i polskiego Twitcha.
6b. Mówisz jako kobieta: wszystkie czasowniki i przymiotniki o sobie w formie żeńskiej (np. „przeanalizowałam”, „sprawdziłam”, „zauważyłam”, „nie udało mi się”, „jestem pewna”), nigdy w męskiej („przeanalizowałem”, „sprawdziłem”).
7. Zacznij od przedstawienia się jako analityczka kanału ${channelName}, powiedz jaki jest dziś dzień, data i godzina. Następnie: ${note}

═══ CO ANALIZOWAĆ ═══
A. Wzorce: gry, dni tygodnia, godziny startu i końca. Najlepsze godziny i dni bierz z "wyliczeniaSerwera" (nie licz w pamięci) i szukaj PRZYCZYN (początek weekendu, święto, wakacje/ferie, pora roku, wydarzenia z kontekstu).
A2. Wzorce WEWNĄTRZ streama: w "przebieg" (punkty mają "godzina") i w słupkach z "aktualnie" znajdź wzrosty i spadki widowni i czatu, podaj godzinę i prawdopodobną przyczynę.
B. Gry: nowsze mają większą wagę (policzona). Wykorzystaj kontekst o oglądalności, stosunku widzów do streamujących, patchach i eventach — wyłącznie jako informację (zasada 3).
C. Tytuł i gra a wyniki: przy „DROP”/„drops” w tytule sprawdź, czy wzrost widowni przełożył się na piszących. Widzów więcej, a czat bez zmian = widzowie głównie „po nagrody”; rosną oba = stream faktycznie trzyma.
D. Korelacje: więcej widzów + więcej wiadomości = dobry znak; mało widzów, ale obecnych długo = stabilna publiczność; mało widzów i pisze ta sama grupa (wysoki udział czatujących) = własne grono. Wskaż, która sytuacja wygląda na prawdziwą.
E. Rajdy: "skokiWidzow" = nagły wzrost z szybkim odpływem. Powiedz, że to prawdopodobnie rajd lub hosting (hipoteza), jak wygląda widownia „bazowa” bez skoku. Bez nicków.
F. Czas trwania: nie porównuj sesji 1:1 — używaj wartości na godzinę z "porownanieNajnowszejSesji" i wspominaj o różnicy długości.
G0. Kiedy wpadają followy: pole "followyKiedy" (liczby followów w ostatnich 30 dniach wg godziny czasu polskiego 0–23 i dnia tygodnia pon–niedz, oraz jaki ułamek wpadł podczas streamów). Wskaż godziny i dni, w których followy wpadają najczęściej, i podaj prawdopodobne przyczyny — wyłącznie jako obserwację (zasada 3), bez rad zmiany pory streamowania. Przy małej próbce (poniżej ok. 20 followów) powiedz, że to za mało danych na wnioski; nie przesadzaj ze znaczeniem pojedynczych liczb. Dane followyKiedy obejmują dokładnie ostatnie 30 dni — nigdy nie mów o 60 dniach.
G. Tempo rozwoju: followersi w 30 dni i wzrost procentowy, saldo po unfollowach, suby (nowe i wygasłe), data ostatniego followa i suba. Nazwij dobry okres, jeśli rosną. Zaznacz, że część danych subów jest przybliżona.
H. Zawsze szukaj PRZYCZYN odchyleń, nie samego „o 5% więcej niż zwykle”.
${modeBlock}

═══ DANE ═══
KONTEKST Z SIECI:
${netBlock}

CZAS I STAN TERAZ:
${J(teraz)}

KANAŁ POZA SESJAMI:
${J(kanal)}

NAJNOWSZA SESJA ("przebieg": min = minuta od startu, v = średnia widzów, msg = wiadomości w odcinku, cz = skumulowani unikalni czatujący):
${J(currentSession)}

HISTORIA SESJI (najnowsze pierwsze):
${J(sessionsHistory)}

WYLICZENIA SERWERA (średnie ważone, normalizacja na godzinę, porównania %):
${J(derived)}

═══ FORMAT ODPOWIEDZI ═══
Zwróć WYŁĄCZNIE gotowy tekst wypowiedzi do przeczytania na głos po polsku — czysty tekst, bez markdownu, wypunktowań, symboli, emoji, nagłówków, JSON-a i adresów stron. Krótkie zdania; liczby i godziny zapisane tak, by dobrze brzmiały na głos. Zachowaj długość z sekcji TRYB.`;
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
        const live = body.tryb === 'live' || (body.tryb == null && !!currentSession?.trwaJeszcze);
        const mode = live ? 'live' : 'offline';

        if (noData) {
            const txt = `Cześć! Tu analityczka kanału ${channelName}. Na razie nie mam zapisanych żadnych zakończonych sesji ani danych kanału, więc nie ma jeszcze czego analizować. Po pierwszej zakończonej transmisji omówię wyniki, a potem szerszy obraz polskiego Twitcha.`;
            return res.json({ success: true, analysis: txt, audioPrompt: txt, searchUsed: false, sources: [], mode });
        }

        const cal = calendarContext();
        // Klucz pamięci podręcznej: te same dane = ta sama analiza (0 zapytań do Gemini)
        const key = live
            ? `A|live|${currentSession?.start}|${currentSession?.glownaGra || teraz.gra || ''}|${teraz.tytul || currentSession?.tytul || ''}`
            : `A|off|${currentSession?.start}|${currentSession?.koniec}|${cal.dzis.dataISO}`;
        if (body.odswiez) cacheMain.del(key);

        const result = await cacheMain.memo(key, live ? TTL.analysisLive : TTL.analysisOffline, async () => {
            const tStart = Date.now();
            const derived = derive(currentSession, sessionsHistory);
            const ctx = await research(body, cal, mode);           // zwykle z pamięci podręcznej
            const searchUsed = !!ctx.net;
            const prompt = buildPrompt({ ...body, live, currentSession: withClock(currentSession), sessionsHistory: sessionsHistory.map(withClock) }, cal, derived, ctx);
            const r = await genSeq(prompt, false, PER_MODEL_MS, THINK[mode], TEXT_MODELS, TEXT_MODELS.length, tStart + TOTAL_BUDGET_MS);
            const parsed = String(r.text).trim().startsWith('{') ? parseJson(r.text) : null;
            const text = cleanSpeech(parsed?.audioPrompt || parsed?.textResponse || r.text);
            if (!text) throw new Error('Pusta odpowiedź modelu');
            // Gdy nie udało się zbadać sieci, nie utrwalamy tej uboższej analizy na długo
            if (!searchUsed) setTimeout(() => cacheMain.del(key), 60e3).unref?.();
            return { text, searchUsed, sources: ctx.sources };
        });

        res.json({ success: true, analysis: result.text, audioPrompt: result.text, searchUsed: result.searchUsed, sources: result.sources || [], mode });
    } catch (error) {
        console.error('Błąd podczas analizy:', error);
        if (isQuota(error)) return res.status(429).json({ success: false, error: 'Limit Gemini wyczerpany — spróbuj za chwilę (limit minutowy) lub po ok. 9:00 (limit dzienny). Stan: ai.dev/rate-limit' + (blockedSummary() ? ' · wstrzymane: ' + blockedSummary() : '') });
        res.status(500).json({ success: false, error: error.message });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Serwer analityczny działa na porcie ${PORT}`));
