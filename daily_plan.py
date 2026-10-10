#!/usr/bin/env python3
"""
Runs once a day in GitHub Actions. Reads Nifty + India VIX, runs the AI/regime model from
nifty50_hedged_bot.py, and writes docs/index.html (your daily plan page) + docs/history.json.
It NEVER places orders. Telegram is used only if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are set.
"""
import json
from pathlib import Path

import pandas as pd
import nifty50_hedged_bot as bot

PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Nifty 50 daily plan</title>
<style>
:root{--bg:#f6f7f4;--card:#fff;--ink:#16202a;--mute:#5d6a75;--line:#d9ded8;--bull:#1f7a4d;--neu:#a8700d;--bear:#b3342b;--acc:#1d4e6b;
box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media(prefers-color-scheme:dark){:root{--bg:#10171d;--card:#18222b;--ink:#e8edf0;--mute:#9aa8b3;--line:#2b3943;--bull:#4cc486;--neu:#e0a73e;--bear:#ef7d72;--acc:#7cb7d6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:720px;margin:0 auto;padding:22px 16px 48px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.v{border-left:6px solid var(--c);}.big{font-size:23px;font-weight:700;color:var(--c)}
.note{color:var(--mute);font-size:13.5px}.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14.5px}
td,th{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--mute);font-weight:600}
input{width:100%;max-width:260px;padding:9px;font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px}
.warn{margin-top:10px;padding:9px 12px;border:1px solid var(--neu);border-radius:8px;font-size:14px}
</style></head><body><main>
<h1>Nifty 50 daily plan</h1>
<p class="note" id="asof"></p>
<div class="card v" id="verdict"></div>

<h2>Market numbers</h2><div class="card wrap" id="nums"></div>

<h2>Your orders</h2>
<div class="card">
<label for="cap"><b>Your capital for this strategy (₹)</b></label><br>
<input id="cap" type="number" inputmode="numeric" placeholder="e.g. 3000000">
<span class="note"> &nbsp;Stays in your browser only.</span>
<div id="orders" class="wrap" style="margin-top:10px"></div>
</div>

<h2>Last 15 days</h2><div class="card wrap" id="hist"></div>

<h2>Rules</h2>
<div class="card"><ul>
<li>Place orders after reading this page, for the next market open.</li>
<li>Roll your put (sell it, buy a new one) when 10 days or fewer remain.</li>
<li>Use real option prices from your broker. Premium here is a model estimate.</li>
<li>This page does not trade for you and is not investment advice.</li></ul>
<p class="note"><a href="backtest.txt">Latest backtest results</a> (if available)</p></div>
</main>
<script>
const P=__PLAN__, H=__HIST__;
const inr=x=>"₹"+Math.round(x).toLocaleString("en-IN"), $=id=>document.getElementById(id);
const col={BULL:"var(--bull)",NEUTRAL:"var(--neu)",BEAR:"var(--bear)"}[P.state]||"var(--acc)";
$("asof").textContent="Based on the close of "+P.date+".";
$("verdict").style.setProperty("--c",col);
$("verdict").innerHTML=`<div class="big">${P.state}: ${Math.round(P.expo*100)}% in Nifty ETF, ${Math.round(P.hedge*100)}% of it insured</div>
<p style="margin:6px 0 0">AI chance of Nifty being higher in ${P.horizon} trading days: <b>${(P.p*100).toFixed(0)}%</b>.${P.changed?" <b>Changed from "+P.prev+".</b>":""}</p>
${P.spike?'<div class="warn">Big market move today. Review before trading.</div>':""}`;
$("nums").innerHTML=`<table><tr><td>Nifty 50</td><td>${P.spot.toLocaleString("en-IN")}</td></tr>
<tr><td>India VIX</td><td>${P.vix.toFixed(1)}</td></tr><tr><td>50-day EMA</td><td>${Math.round(P.ema50).toLocaleString("en-IN")}</td></tr>
<tr><td>200-day average</td><td>${Math.round(P.ma200).toLocaleString("en-IN")}</td></tr></table>`;
$("hist").innerHTML="<table><tr><th>Date</th><th>State</th><th>AI chance</th><th>Nifty</th><th>VIX</th></tr>"+
H.slice().reverse().map(h=>`<tr><td>${h.date}</td><td>${h.state}</td><td>${Math.round(h.p*100)}%</td><td>${Math.round(h.spot).toLocaleString("en-IN")}</td><td>${h.vix.toFixed(1)}</td></tr>`).join("")+"</table>";
try{const c=localStorage.getItem("cap");if(c)$("cap").value=c;}catch(e){}
function orders(){
  const cap=parseFloat($("cap").value);try{localStorage.setItem("cap",$("cap").value);}catch(e){}
  if(!(cap>0)){$("orders").innerHTML='<p class="note">Enter your capital to see units and lots.</p>';return;}
  let units=Math.floor(cap*P.expo/P.etf),lots=0,trim=false;
  if(units>0&&P.hedge>0){lots=Math.round(P.hedge*units*P.etf/(P.lot*P.spot));if(lots===0){units=Math.floor(units*(1-.25*P.hedge));trim=true;}}
  let t=`<table><tr><td>Hold NIFTYBEES</td><td><b>${units.toLocaleString("en-IN")} units</b> (${inr(units*P.etf)})</td></tr>`;
  if(lots>0)t+=`<tr><td>Hold Nifty put</td><td><b>${lots} lot(s)</b>, strike <b>${P.strike.toLocaleString("en-IN")}</b>, about ${P.dte} days to expiry</td></tr>
  <tr><td>Estimated cost</td><td>about ${inr(P.prem*P.lot*lots)} (${P.prem.toFixed(0)} per unit; lot size ${P.lot})</td></tr>`;
  else if(P.state==="BEAR")t+=`<tr><td>Put</td><td>None. Close any put you hold.</td></tr>`;
  $("orders").innerHTML=t+"</table>"+(trim?`<div class="warn">Capital is too small for one Nifty lot (about ${inr(P.lot*P.spot)}). ETF was trimmed instead of buying a put.</div>`:"")+
  `<p class="note">Compare with what you already hold and trade only the difference.</p>`;
}
$("cap").addEventListener("input",orders);orders();
</script></body></html>"""


def main():
    cfg = bot.Config()
    df, vix = bot.fetch("^NSEI", 12), bot.fetch("^INDIAVIX", 12).Close
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    if df.index[-1] == now.tz_localize(None).normalize() and now.hour * 60 + now.minute < 16 * 60:
        df = df.iloc[:-1]                                          # ignore unfinished daily bar

    X, ema50, ma200 = bot.build_features(df, vix)
    y = (df.Close.shift(-cfg.horizon) / df.Close - 1 > 0).astype(int)
    Xv = X.dropna()
    n = len(Xv) - cfg.horizon
    model = bot.new_model().fit(Xv.iloc[:n], y.loc[Xv.index].iloc[:n])
    p = float(model.predict_proba(Xv.iloc[[-1]])[:, 1][0])
    t = Xv.index[-1]
    spot, v = float(df.Close[t]), float(vix.reindex(df.index).ffill()[t])
    state, expo, hedge = bot.decide(p, spot, ema50[t], ma200[t], v, cfg)

    try:
        etf = float(bot.fetch("NIFTYBEES", 1).Close.iloc[-1])
    except Exception:
        etf = spot / 100
    K = int(round(spot * (1 - cfg.put_otm) / 50) * 50)
    prem = bot.bs_put(spot, K, cfg.put_dte_buy / 365, cfg.r, v / 100 * (1 + cfg.skew))
    ret1 = float(df.Close.pct_change().iloc[-1]); vix1 = float(vix.pct_change().iloc[-1])

    out = Path("docs"); out.mkdir(exist_ok=True)
    hp = out / "history.json"
    hist = json.loads(hp.read_text()) if hp.exists() else []
    date = t.date().isoformat()
    prev = next((h["state"] for h in reversed(hist) if h["date"] != date), None)
    hist = [h for h in hist if h["date"] != date] + [dict(date=date, state=state, p=round(p, 3), spot=spot, vix=v)]
    hist = hist[-250:]
    hp.write_text(json.dumps(hist))

    plan = dict(date=date, state=state, expo=expo, hedge=hedge, p=p, horizon=cfg.horizon, spot=spot, vix=v,
                ema50=float(ema50[t]), ma200=float(ma200[t]), etf=etf, strike=K, prem=float(prem),
                lot=cfg.lot_size, dte=cfg.put_dte_buy, prev=prev, changed=bool(prev and prev != state),
                spike=bool(abs(ret1) > 0.03 or vix1 > 0.20))
    (out / "index.html").write_text(PAGE.replace("__PLAN__", json.dumps(plan)).replace("__HIST__", json.dumps(hist[-15:])))

    msg = (f"📊 Nifty50 plan {date}: {state}\nNifty {spot:,.0f} | VIX {v:.1f} | AI P(up) {p:.0%}\n"
           f"Hold {expo:.0%} NIFTYBEES, hedge {hedge:.0%} with put strike {K}"
           + (f"\n🔔 Changed from {prev}" if plan["changed"] else "") + ("\n🚨 Big move today" if plan["spike"] else ""))
    print(msg)
    bot.tg(msg)                      # silently skipped unless Telegram secrets are set


if __name__ == "__main__":
    main()
