#!/usr/bin/env python3
"""
Nifty-50 INDEX bot: long NIFTYBEES (ETF) + dynamic protective-put hedge on Nifty options.
Telegram alerts for every run, state change, order, and error.

  python nifty50_hedged_bot.py backtest [--years 12]     # compares 4 strategies (see below)
  python nifty50_hedged_bot.py signals                   # dry run: what it WOULD do today
  python nifty50_hedged_bot.py signals --execute         # paper orders
  python nifty50_hedged_bot.py signals --execute --live  # REAL orders (Zerodha Kite, AMO limit)
  python nifty50_hedged_bot.py telegram-test | halt | resume

Install: pip install yfinance pandas numpy scikit-learn kiteconnect
Env:     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BOT_CAPITAL
         (live) KITE_API_KEY, KITE_ACCESS_TOKEN, I_ACCEPT_REAL_MONEY_RISK=yes

Educational software, not investment advice. Run the backtest, paper trade, then go small.
"""
import argparse, json, logging, math, os, sys, traceback, urllib.parse, urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")


@dataclass
class Config:
    # model / regime
    horizon: int = 10
    enter_p: float = 0.55
    exit_p: float = 0.45
    min_train: int = 750
    retrain_every: int = 60
    # exposure (fraction of capital in NIFTYBEES) and hedge ratio (fraction of that exposure covered by puts)
    expo_bull: float = 1.0;  expo_neutral: float = 0.5;  expo_bear: float = 0.0
    hedge_bull: float = 0.5; hedge_neutral: float = 1.0
    hedge_bull_when_vix_high: float = 0.25   # puts are expensive when India VIX is high
    vix_high: float = 22.0
    # put contract
    put_otm: float = 0.05        # strike = 95% of spot
    put_dte_buy: int = 42        # buy ~6-week put
    put_roll_dte: int = 10       # roll when <=10 days left
    lot_size: int = 65           # NSE Nifty lot size since Jan 2026; live mode reads it from Kite
    # backtest pricing assumptions (Black-Scholes, India VIX as IV)
    r: float = 0.065
    skew: float = 0.10           # OTM put IV = VIX * (1 + skew)
    spread: float = 0.02         # option bid-ask cost, fraction of premium per trade
    etf_cost: float = 0.0007     # ETF brokerage/charges/slippage per unit turnover
    cash_yield: float = 0.06     # idle cash in liquid fund
    # live
    capital: float = 500_000.0
    put_delta_trim: float = 0.25 # if capital too small for 1 lot, trim ETF by delta*ratio instead
    state_file: str = "hedged_bot_state.json"


# ============================================================================ Telegram
def tg(msg: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.info("[Telegram not configured]\n%s", msg)
        return False
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg[:4000]}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=15)
        return True
    except Exception as e:                       # never let alerting crash trading
        log.warning("Telegram send failed: %s", type(e).__name__)
        return False


# ============================================================================ data & features
def fetch(symbol: str, years: int = 12) -> pd.DataFrame:
    import yfinance as yf
    t = symbol if symbol.startswith("^") else f"{symbol}.NS"
    df = yf.Ticker(t).history(period=f"{years}y", auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(subset=["Close"])


def build_features(df: pd.DataFrame, vix: pd.Series):
    c, h, l = df.Close, df.High, df.Low
    v = vix.reindex(df.index).ffill()
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20, 60):
        f[f"ret{n}"] = c.pct_change(n)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-9))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ema20, ema50, ma200 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean(), c.rolling(200).mean()
    f["ema_gap"] = ema20 / ema50 - 1
    f["dist200"] = c / ma200 - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - ma) / (2 * sd + 1e-9)
    f["vol20"] = f["ret1"].rolling(20).std()
    f["dd_52w"] = c / c.rolling(250, min_periods=60).max() - 1
    f["vix"] = v
    f["vix_chg5"] = v.pct_change(5)
    f["vix_rel"] = v / v.rolling(250, min_periods=60).mean() - 1
    return f.replace([np.inf, -np.inf], np.nan), ema50, ma200


def new_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, random_state=7)


def decide(p, close, ema50, ma200, vix, cfg: Config):
    """Return (state, exposure, hedge_ratio)."""
    if p > cfg.enter_p and close > ema50 and close > ma200:
        return "BULL", cfg.expo_bull, (cfg.hedge_bull if vix <= cfg.vix_high else cfg.hedge_bull_when_vix_high)
    if p < cfg.exit_p and close < ma200:
        return "BEAR", cfg.expo_bear, 0.0
    return "NEUTRAL", cfg.expo_neutral, cfg.hedge_neutral


# ============================================================================ options pricing (backtest)
def _N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)


def put_overlay(close: pd.Series, vix: pd.Series, cfg: Config):
    """Rolling OTM put, per 1 unit of hedged notional. Returns (daily pnl/notional, premium/spot)."""
    idx = close.index
    u, prem = pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)
    K, exp, prev = None, None, 0.0
    for i, t in enumerate(idx):
        S = float(close.iat[i]); sig = float(vix.iat[i]) / 100 * (1 + cfg.skew)
        if K is not None:
            dte = (exp - t).days
            val = bs_put(S, K, max(dte, 0) / 365, cfg.r, sig)
            u.iat[i] = (val - prev) / float(close.iat[i - 1]); prev = val
            if dte <= cfg.put_roll_dte:                      # roll
                u.iat[i] -= cfg.spread * val / S; K = None
        if K is None:
            K, exp = S * (1 - cfg.put_otm), t + pd.Timedelta(days=cfg.put_dte_buy)
            prev = bs_put(S, K, cfg.put_dte_buy / 365, cfg.r, sig)
            u.iat[i] -= cfg.spread * prev / S
        prem.iat[i] = prev / S
    return u, prem


# ============================================================================ backtest
def walk_forward(X, y, cfg):
    proba = pd.Series(np.nan, index=X.index)
    for i in range(cfg.min_train, len(X), cfg.retrain_every):
        end = i - cfg.horizon                        # purge so labels are known at time i
        m = new_model().fit(X.iloc[:end], y.iloc[:end])
        j = min(i + cfg.retrain_every, len(X))
        proba.iloc[i:j] = m.predict_proba(X.iloc[i:j])[:, 1]
    return proba


def backtest(df: pd.DataFrame, vix: pd.Series, cfg: Config) -> pd.DataFrame:
    X, ema50, ma200 = build_features(df, vix)
    y = (df.Close.shift(-cfg.horizon) / df.Close - 1 > 0).astype(int)
    Xv = X.dropna()
    proba = walk_forward(Xv, y.loc[Xv.index], cfg).dropna()
    idx = proba.index
    d, v = df.loc[idx], vix.reindex(idx).ffill()
    st = [decide(proba[t], d.Close[t], ema50[t], ma200[t], v[t], cfg) for t in idx]
    expo = pd.Series([s[1] for s in st], index=idx)
    hedge = pd.Series([s[2] for s in st], index=idx)

    gap = d.Open / d.Close.shift(1) - 1             # overnight (AMO orders fill at next open)
    intra = d.Close / d.Open - 1
    u, prem = put_overlay(d.Close, v, cfg)

    def run(e, h):
        e1, e2 = e.shift(1).bfill(), e.shift(2).bfill()
        p1, p2 = (e * h).shift(1).bfill(), (e * h).shift(2).bfill()
        r = (e2 * gap.fillna(0) + e1 * intra + (1 - e1) * cfg.cash_yield / 252
             - cfg.etf_cost * (e1 - e2).abs() + p1 * u - cfg.spread * prem * (p1 - p2).abs())
        return r.iloc[1:]

    one, zero = pd.Series(1.0, index=idx), pd.Series(0.0, index=idx)
    out = pd.DataFrame({
        "1 Buy & hold":            run(one, zero),
        "2 Buy & hold + puts":     run(one, one),
        "3 AI-timed, no hedge":    run(expo, zero),
        "4 AI-timed + dyn. hedge": run(expo, hedge),
    })
    out.attrs["state_mix"] = pd.Series([s[0] for s in st]).value_counts(normalize=True).round(2).to_dict()
    return out


def report(rets: pd.DataFrame):
    log.info("%-26s %7s %7s %7s %8s %7s", "Strategy", "CAGR%", "Vol%", "Sharpe", "MaxDD%", "Calmar")
    for name, r in rets.items():
        eq = (1 + r).cumprod(); yrs = len(r) / 252
        cagr = eq.iloc[-1] ** (1 / yrs) - 1; vol = r.std() * np.sqrt(252)
        mdd = (eq / eq.cummax() - 1).min()
        log.info("%-26s %7.1f %7.1f %7.2f %8.1f %7.2f", name, cagr * 100, vol * 100,
                 np.sqrt(252) * r.mean() / (r.std() + 1e-12), mdd * 100, cagr / abs(mdd) if mdd else 0)
    log.info("Time in states: %s", rets.attrs.get("state_mix"))


# ============================================================================ brokers
class PaperBroker:
    live = False
    def etf_ltp(self): return float(fetch("NIFTYBEES", 1).Close.iloc[-1])
    def etf_qty(self): return None
    def find_put(self, spot, vix, cfg):
        today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        exp = (today + pd.Timedelta(days=cfg.put_dte_buy)).date()
        K = int(round(spot * (1 - cfg.put_otm) / 50) * 50)
        px = bs_put(spot, K, cfg.put_dte_buy / 365, cfg.r, vix / 100 * (1 + cfg.skew))
        return dict(symbol=f"NIFTY{exp:%d%b%y}{K}PE".upper(), strike=K, expiry=exp.isoformat(),
                    lot_size=cfg.lot_size, ltp=round(px, 2))
    def put_ltp(self, put, spot, vix, cfg):
        dte = (pd.Timestamp(put["expiry"]).date() - pd.Timestamp.now(tz="Asia/Kolkata").date()).days
        return max(bs_put(spot, put["strike"], max(dte, 0) / 365, cfg.r, vix / 100 * (1 + cfg.skew)), 0.05)
    def place(self, o):
        log.info("[PAPER] %s %s x%d @ ~%.2f", o["side"], o["symbol"], o["qty"], o["px"])
        with open("paper_orders.csv", "a") as f:
            f.write(f'{pd.Timestamp.now()},{o["symbol"]},{o["side"]},{o["qty"]},{o["px"]:.2f}\n')


class KiteBroker(PaperBroker):
    """Zerodha Kite Connect. ETF: NSE/CNC. Puts: NFO/NRML. All AMO LIMIT orders (fill next open)."""
    live = True
    def __init__(self):
        from kiteconnect import KiteConnect
        self.k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
        self.k.set_access_token(os.environ["KITE_ACCESS_TOKEN"])
    def etf_ltp(self): return float(self.k.ltp(["NSE:NIFTYBEES"])["NSE:NIFTYBEES"]["last_price"])
    def etf_qty(self):
        return sum(int(h["quantity"]) + int(h.get("t1_quantity", 0))
                   for h in self.k.holdings() if h["tradingsymbol"] == "NIFTYBEES")
    def find_put(self, spot, vix, cfg):
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        ins = [i for i in self.k.instruments("NFO")
               if i["name"] == "NIFTY" and i["instrument_type"] == "PE" and (i["expiry"] - today).days >= 25]
        exp = min({i["expiry"] for i in ins}, key=lambda e: abs((e - today).days - cfg.put_dte_buy))
        tgt = spot * (1 - cfg.put_otm)
        i = min((i for i in ins if i["expiry"] == exp), key=lambda i: abs(i["strike"] - tgt))
        ltp = self.k.ltp([f'NFO:{i["tradingsymbol"]}'])[f'NFO:{i["tradingsymbol"]}']["last_price"]
        return dict(symbol=i["tradingsymbol"], strike=i["strike"], expiry=exp.isoformat(),
                    lot_size=int(i["lot_size"]), ltp=float(ltp))
    def put_ltp(self, put, spot, vix, cfg):
        s = f'NFO:{put["symbol"]}'
        return float(self.k.ltp([s])[s]["last_price"])
    def place(self, o):
        k = self.k
        oid = k.place_order(variety=k.VARIETY_AMO, exchange=o["exchange"], tradingsymbol=o["symbol"],
                            transaction_type=o["side"], quantity=o["qty"],
                            product=k.PRODUCT_CNC if o["exchange"] == "NSE" else k.PRODUCT_NRML,
                            order_type=k.ORDER_TYPE_LIMIT, price=o["px"])
        log.info("[LIVE] %s %s x%d @ %.2f -> %s", o["side"], o["symbol"], o["qty"], o["px"], oid)


# ============================================================================ live signals
def load_state(cfg):
    p = Path(cfg.state_file)
    s = json.loads(p.read_text()) if p.exists() else {}
    return {"state": None, "etf_qty": 0, "put": None, "halted": False, **s}


def save_state(cfg, s): Path(cfg.state_file).write_text(json.dumps(s, indent=2))


def limit_px(ltp, side):                       # marketable-ish limit, tick-rounded
    return round(round(ltp * (1.005 if side == "BUY" else 0.995) / 0.05) * 0.05, 2)


def run_signals(a, cfg: Config):
    mode = "LIVE" if (a.execute and a.live) else "PAPER" if a.execute else "DRY-RUN"
    if mode == "LIVE" and os.environ.get("I_ACCEPT_REAL_MONEY_RISK") != "yes":
        raise SystemExit("Set I_ACCEPT_REAL_MONEY_RISK=yes to enable live orders.")
    broker = KiteBroker() if mode == "LIVE" else PaperBroker()
    S = load_state(cfg)
    capital = float(os.environ.get("BOT_CAPITAL", cfg.capital))

    df, vix = fetch("^NSEI", 12), fetch("^INDIAVIX", 12).Close
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    if df.index[-1] == now.tz_localize(None).normalize() and now.hour * 60 + now.minute < 16 * 60:
        df = df.iloc[:-1]                                      # drop partial intraday bar
    X, ema50, ma200 = build_features(df, vix)
    y = (df.Close.shift(-cfg.horizon) / df.Close - 1 > 0).astype(int)
    Xv = X.dropna()
    model = new_model().fit(Xv.iloc[:len(Xv) - cfg.horizon], y.loc[Xv.index].iloc[:len(Xv) - cfg.horizon])
    p = float(model.predict_proba(Xv.iloc[[-1]])[:, 1][0])
    t = Xv.index[-1]
    spot, v = float(df.Close[t]), float(vix.reindex(df.index).ffill()[t])
    state, expo, hedge = decide(p, spot, ema50[t], ma200[t], v, cfg)
    if S["halted"]:
        state, expo, hedge = "HALTED", 0.0, 0.0
    ret1 = float(df.Close.pct_change().iloc[-1]); vix1 = float(vix.pct_change().iloc[-1])

    # ---- targets
    etf_px = broker.etf_ltp()
    cur_etf = broker.etf_qty() if broker.live else S["etf_qty"]
    if broker.live and cur_etf != S["etf_qty"]:
        tg(f"⚠️ ETF qty mismatch: state={S['etf_qty']} broker={cur_etf}. Using broker.")
    tgt_etf = int(capital * expo / etf_px)
    put = broker.find_put(spot, v, cfg) if (hedge > 0 and tgt_etf > 0) else None
    lots = 0
    note = ""
    if put:
        lots = round(hedge * tgt_etf * etf_px / (put["lot_size"] * spot))
        if lots == 0:                                          # too small for 1 lot -> trim instead
            tgt_etf = int(tgt_etf * (1 - cfg.put_delta_trim * hedge)); put = None
            note = "Capital < 1 Nifty lot of hedge: trimmed ETF by put-delta instead of buying puts."

    # ---- orders
    orders, cur = [], S["put"]
    d = tgt_etf - cur_etf
    if d != 0:
        side = "BUY" if d > 0 else "SELL"
        orders.append(dict(symbol="NIFTYBEES", exchange="NSE", side=side, qty=abs(d), px=limit_px(etf_px, side), kind="ETF"))
    new_put = None
    if cur:
        dte = (pd.Timestamp(cur["expiry"]).date() - now.date()).days
        px_cur = broker.put_ltp(cur, spot, v, cfg)
        if dte <= cfg.put_roll_dte or lots == 0:
            orders.append(dict(symbol=cur["symbol"], exchange="NFO", side="SELL", qty=cur["lots"] * cur["lot_size"],
                               px=limit_px(px_cur, "SELL"), kind="PUT"))
        elif lots != cur["lots"]:
            ds = lots - cur["lots"]; side = "BUY" if ds > 0 else "SELL"
            orders.append(dict(symbol=cur["symbol"], exchange="NFO", side=side, qty=abs(ds) * cur["lot_size"],
                               px=limit_px(px_cur, side), kind="PUT"))
            new_put = {**cur, "lots": lots}
        else:
            new_put = cur
    if lots > 0 and new_put is None:
        orders.append(dict(symbol=put["symbol"], exchange="NFO", side="BUY", qty=lots * put["lot_size"],
                           px=limit_px(put["ltp"], "BUY"), kind="PUT"))
        new_put = {"symbol": put["symbol"], "strike": put["strike"], "expiry": put["expiry"],
                   "lots": lots, "lot_size": put["lot_size"]}

    # ---- alert
    prev = S["state"]
    lines = [f"📊 Nifty50 Hedged Bot — {now:%d %b %Y} [{mode}]",
             f"Nifty {spot:,.0f} ({ret1:+.2%}) | India VIX {v:.1f} ({vix1:+.1%})",
             f"AI P(up {cfg.horizon}d) = {p:.2f} | 50EMA {ema50[t]:,.0f} | 200DMA {ma200[t]:,.0f}",
             f"Regime: {state}" + (f"  🔔 changed from {prev}" if prev and prev != state else ""),
             f"Target: {expo:.0%} NIFTYBEES ({tgt_etf} units), hedge {hedge:.0%} → {lots} put lot(s)"
             + (f" [{put['symbol']}]" if put else "")]
    if note: lines.append("ℹ️ " + note)
    if abs(ret1) > 0.03 or vix1 > 0.20: lines.append("🚨 VOLATILITY SPIKE — consider reviewing manually.")
    lines += ["Orders:"] + ([f"• {o['side']} {o['qty']} {o['symbol']} @ {o['px']:.2f}" for o in orders] or ["• none"])
    if mode == "LIVE": lines.append("Orders are AMO limit orders — verify fills in Kite tomorrow.")
    msg = "\n".join(lines); log.info("\n%s", msg)
    tg(msg)

    if a.execute:
        for o in orders:
            broker.place(o)
        S.update(state=state, etf_qty=tgt_etf, put=new_put)
        save_state(cfg, S)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest"); b.add_argument("--years", type=int, default=12)
    s = sub.add_parser("signals"); s.add_argument("--execute", action="store_true"); s.add_argument("--live", action="store_true")
    for c in ("telegram-test", "halt", "resume"): sub.add_parser(c)
    a, cfg = ap.parse_args(), Config()
    try:
        if a.cmd == "backtest":
            r = backtest(fetch("^NSEI", a.years), fetch("^INDIAVIX", a.years).Close, cfg)
            report(r); r.to_csv("hedged_backtest_returns.csv")
        elif a.cmd == "signals":
            run_signals(a, cfg)
        elif a.cmd == "telegram-test":
            print("sent" if tg("✅ Nifty bot Telegram connection works.") else "NOT sent — check env vars")
        else:
            S = load_state(cfg); S["halted"] = a.cmd == "halt"; save_state(cfg, S)
            tg("⛔ Bot HALTED: next run goes to cash and closes hedges." if S["halted"] else "▶️ Bot resumed.")
    except SystemExit:
        raise
    except Exception:
        tg("🚨 Bot ERROR:\n" + traceback.format_exc()[-1500:]); raise


if __name__ == "__main__":
    main()
