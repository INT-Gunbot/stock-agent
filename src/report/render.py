"""日报渲染：facts -> Markdown / HTML。

渲染层不允许做任何计算，只做排版和格式化。
所以这里看到的每个数字，都能在 daily.py 里找到对应的计算过程。
"""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

from ..config import REPORT_DIR

# 中国习惯：涨红跌绿
UP = "#d93a2b"
DOWN = "#12946a"
NEUTRAL = "#6b7280"


def _cls(v) -> str:
    if v is None:
        return "flat"
    return "up" if v > 0 else "down" if v < 0 else "flat"


def _sgn(v, digits=2, suffix="%") -> str:
    if v is None:
        return "—"
    return f"{v:+.{digits}f}{suffix}"


def _num(v, digits=2) -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def _money(v) -> str:
    """元 -> 万元/亿元。"""
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e8:
        return f"{v/1e8:,.2f} 亿"
    if a >= 1e4:
        return f"{v/1e4:,.1f} 万"
    return f"{v:,.0f}"


# ------------------------------------------------------------------ Markdown
def render_markdown(f: dict) -> str:
    m, mk, se, ev, ho, te, wa = (
        f["meta"], f["market"], f["sentiment"], f["events"],
        f["holdings"], f["technical"], f["watch"],
    )
    L = []
    a = L.append

    a(f"# 交易日报 {m['trade_date']}")
    a("")
    regime = mk.get("regime", {})
    a(f"> 市场档位 **{regime.get('level','—')}** ｜ 情绪分 **{se['score']['score']:.0f} "
      f"({se['score']['label']})** ｜ 建议仓位上限 {regime.get('position_cap','—')}%")
    a(f"> {regime.get('reason','')}")
    a("")
    sm = f.get("summary", {}).get("lines") or []
    if sm:
        a("**今日要点**")
        a("")
        for ln in sm:
            a(f"- {ln}")
        a("")

    # 1 市场概览
    a("## 一、市场概览")
    a("")
    a("| 指数 | 收盘 | 涨跌 | 成交额 | 额环比 | MA20 | MA60 | 20日 |")
    a("|---|---:|---:|---:|---:|:--:|:--:|---:|")
    for i in mk["indices"]:
        a(f"| {i['name']} | {i['close']:,.2f} | {_sgn(i['pct'])} | "
          f"{i['amount_yi']:,.0f} 亿 | {_sgn(i['amount_chg'],1)} | "
          f"{'上方' if i['above_ma20'] else '下方'} | "
          f"{'上方' if i['above_ma60'] else '下方'} | {_sgn(i['ret20'],1)} |")
    b = mk["breadth"]
    a("")
    a(f"- 涨跌家数：**{b['up']} 涨 / {b['down']} 跌 / {b['flat']} 平**，"
      f"上涨占比 {b['up_ratio']:.1f}%，中位数 {_sgn(b['median_pct'])}")
    a(f"- 涨跌停：涨停 **{b['limit_up']}** 家，跌停 **{b['limit_down']}** 家")
    a("")

    # 2 资金与情绪
    a("## 二、资金与情绪")
    a("")
    a(f"- 全市场成交额 **{se['amount_yi']:,.0f} 亿**，环比 {_sgn(se['amount_chg'],1)}，"
      f"较 20 日均值 {_sgn(se['amount_vs_ma20'],1)}，处于近 60 日 {se['amount_pct_rank60']:.0f}% 分位")
    a(f"- 涨超 5%：**{se['up5']}** 家；跌超 5%：**{se['down5']}** 家")
    a(f"- 触及涨停 {se['touched_up']} 家，封板率 **{se['seal_rate']:.0f}%**"
      if se.get("seal_rate") is not None else "- 封板率：—")
    sp = se["score"]["parts"]
    a(f"- 情绪分 **{se['score']['score']:.0f}/100**（{se['score']['label']}）："
      f"广度 {sp['breadth']:+.1f}、涨停 {sp['limit_up']:+.1f}、"
      f"跌停 {sp['limit_down']:+.1f}、封板 {sp['seal']:+.1f}")
    a("")
    a("<details><summary>本区块暂缺的数据</summary>")
    a("")
    for x in se["missing"]:
        a(f"- {x}")
    a("")
    a("</details>")
    a("")

    # 3 事件与消息
    a("## 三、事件与消息")
    a("")
    if not ev["events"]:
        a("今日未从行情与财务数据中检测到可验证事件。")
    for e in ev["events"]:
        a(f"**[{e['type']}] {e['title']}**")
        a("")
        if e.get("detail"):
            a(f"> {e['detail']}")
            a("")
        for it in e.get("items", []):
            a(f"- {it['name']}：{it['value']}")
        a("")

    # 4 持仓异动
    a("## 四、持仓异动")
    a("")
    s = ho.get("summary", {})
    if s:
        a(f"- 组合市值 **{_money(s['mv'])}**，浮盈 **{_money(s['pnl'])}**（{_sgn(s['pnl_pct'])}）")
        a(f"- 当日盈亏 **{_money(s['day_pnl'])}**（{_sgn(s['day_ret'])}），"
          f"相对 {s['bench_name']}（{_sgn(s['bench_ret'])}）超额 **{_sgn(s['excess'])}**")
        a(f"- 盈利 {s['winners']} 只 / 亏损 {s['losers']} 只；"
          f"站上 MA20 {s['above_ma20']}/{s['positions']} 只、MA60 {s['above_ma60']}/{s['positions']} 只")
        a(f"- 前三大权重合计 {s['top3_weight']:.1f}%，最大单票 {s['max_weight_name']} {s['max_weight']:.1f}%")
        a("")
    if ho["alerts"]:
        a(f"**触发阈值的 {len(ho['alerts'])} 只：**")
        a("")
        for x in ho["alerts"]:
            head = (f"**{x['name']}**（{x['ts_code']}） {x['close']:.2f} "
                    f"{_sgn(x['day_ret'])}")
            if x.get("excess") is not None:
                head += f"，超额 {_sgn(x['excess'])}"
            a(head)
            for r in x["reasons"]:
                a(f"  - {r}")
        a("")
    else:
        a("今日无持仓触发异动阈值。")
        a("")
    if ho["quiet"]:
        a(f"其余 {len(ho['quiet'])} 只无异常："
          + "、".join(f"{q['name']}({_sgn(q['day_ret'])})" for q in ho["quiet"]))
        a("")
    if ho.get("table"):
        a("**全部持仓**")
        a("")
        a("| 持仓 | 现价 | 当日 | 权重 | 浮盈 | MA20 | MA60 | 20日位置 | 20日涨跌 |")
        a("|---|---:|---:|---:|---:|:--:|:--:|---:|---:|")
        for t in ho["table"]:
            a(f"| {t['name']} | {t['close']:.2f} | {_sgn(t['day_ret'])} | "
              f"{t['weight']:.1f}% | {_sgn(t['pnl_pct'])} | {t['ma20_side']} | "
              f"{t['ma60_side']} | {_num(t['pos20'])} | {_sgn(t['ret20'],1)} |")
        a("")

    # 5 技术面
    a("## 五、技术面观察")
    a("")
    a("| 标的 | 收盘 | MA20 | MA60 | 20日位置 | ATR | 20日涨跌 | 60日涨跌 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for i in te["indices"]:
        a(f"| {i['name']} | {i['close']:,.2f} | {_num(i['ma20'])} | {_num(i['ma60'])} | "
          f"— | — | {_sgn(i['ret20'],1)} | — |")
    for h in te["holdings"]:
        a(f"| {h['name']} | {h['close']:,.2f} | {_num(h['ma20'])} | {_num(h['ma60'])} | "
          f"{_num(h['pos20'])} | {_num(h['atr_pct'],1)}% | {_sgn(h['ret20'],1)} | "
          f"{_sgn(h['ret60'],1)} |")
    a("")
    hl = te.get("market", {})
    if hl:
        a(f"- 全市场创 60 日新高 **{hl['new_high60']}** 家、新低 **{hl['new_low60']}** 家"
          f"（新高占比 {hl['high_ratio']}%）")
        a("")

    # 6 明日观察
    a("## 六、明日观察清单")
    a("")
    a("条件框架，不构成方向预测：")
    a("")
    a("| 对象 | 若发生 | 则 |")
    a("|---|---|---|")
    last = None
    for w in wa:
        scope = w["scope"] if w["scope"] != last else ""
        last = w["scope"]
        a(f"| {scope} | {w['cond']} | {w['then']} |")
    a("")

    a("---")
    a("")
    a("**数据说明**")
    a(f"- 生成时间 {m['generated_at']}；行情截至 {m['trade_date']}")
    a(f"- 异动阈值：单日涨跌 {m['thresholds']['ret']}%、"
      f"量能比 {m['thresholds']['vol_ratio']}、ATR 倍数 {m['thresholds']['atr']}")
    a("- 涨跌幅已按复权口径修正，除权日不会出现失真的暴跌数字")
    a("- 持仓收益按原始价与持仓成本计算，与券商账户浮动盈亏口径一致")
    for g in m["data_gaps"]:
        a(f"- ⚠️ {g}")
    a("")
    a(DISCLAIMER_MD)
    return "\n".join(L)


DISCLAIMER_MD = (
    "> **免责声明**：本报告由程序基于公开数据自动生成，仅供研究参考，"
    "不构成投资建议。市场有风险，投资需谨慎。任何投资决策应结合个人风险承受能力、"
    "资金状况和投资目标独立判断，必要时咨询持牌专业机构。过往表现不预示未来收益。"
)


# ------------------------------------------------------------------ HTML
CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f5f6f8;color:#1f2430;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
 font-size:14px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
.head{background:#fff;border-radius:14px;padding:26px 28px;box-shadow:0 1px 3px rgba(16,24,40,.06);
 border:1px solid #e8eaee;margin-bottom:18px}
.head h1{margin:0 0 4px;font-size:26px;letter-spacing:-.4px}
.head .sub{color:#7c8494;font-size:13px}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.badge{padding:6px 14px;border-radius:999px;font-size:13px;font-weight:600;border:1px solid}
.b-regime{background:#eef4ff;border-color:#cfe0ff;color:#2559c9}
.b-score{background:#fff4e8;border-color:#ffdcb8;color:#b25c00}
.b-cap{background:#f0f7f2;border-color:#cfe8d8;color:#1c7a4a}
.reason{margin-top:12px;color:#5d6675;font-size:13px}
.points{margin-top:16px;padding-top:16px;border-top:1px solid #eef0f4;display:flex;flex-direction:column;gap:8px}
.pt{position:relative;padding-left:16px;color:#333b4a;font-size:13.5px;line-height:1.7}
.pt:before{content:"";position:absolute;left:0;top:9px;width:5px;height:5px;border-radius:50%;background:#c3cad6}
.card{background:#fff;border-radius:14px;padding:22px 24px;margin-bottom:16px;
 border:1px solid #e8eaee;box-shadow:0 1px 3px rgba(16,24,40,.05)}
.card h2{margin:0 0 4px;font-size:17px;display:flex;align-items:center;gap:8px}
.card h2 .n{display:inline-flex;width:22px;height:22px;border-radius:6px;background:#eef1f6;
 color:#5d6675;font-size:12px;align-items:center;justify-content:center;font-weight:700}
.card .hint{color:#8b93a3;font-size:12px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:right;color:#8b93a3;font-weight:600;padding:8px 10px;border-bottom:1px solid #eceef2;
 font-size:12px;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:9px 10px;border-bottom:1px solid #f2f4f7;text-align:right;white-space:nowrap}
tr:last-child td{border-bottom:none}
.up{color:#d93a2b;font-weight:600}
.down{color:#12946a;font-weight:600}
.flat{color:#6b7280}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:6px 0 16px}
.kpi{background:#fafbfc;border:1px solid #eceef2;border-radius:10px;padding:12px 14px}
.kpi .k{color:#8b93a3;font-size:12px}
.kpi .v{font-size:19px;font-weight:700;margin-top:2px;letter-spacing:-.3px}
.kpi .x{color:#9aa1b0;font-size:11px;margin-top:2px}
.bar{height:8px;background:#eceef2;border-radius:4px;overflow:hidden;margin-top:8px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#ffb066,#d93a2b)}
.ev{border-left:3px solid #e2e6ee;padding:2px 0 2px 14px;margin:14px 0}
.ev.warn{border-left-color:#f0a020}
.ev .t{font-weight:700;margin-bottom:4px}
.ev .d{color:#6f7787;font-size:13px;margin-bottom:8px}
.ev ul{margin:6px 0 0;padding-left:18px;color:#414a5a}
.ev li{margin:3px 0}
.alert{border:1px solid #eceef2;border-radius:10px;padding:13px 15px;margin-bottom:10px;background:#fcfcfd}
.alert.s2{border-left:3px solid #e8892b}
.alert.s3{border-left:3px solid #d93a2b}
.alert .h{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.alert .nm{font-weight:700}
.tag{display:inline-block;font-size:11px;color:#8b93a3;background:#f0f2f6;border-radius:5px;
 padding:1px 6px;margin-left:6px;font-weight:400;vertical-align:middle}
.alert ul{margin:8px 0 0;padding-left:18px;color:#4a5261;font-size:13px}
.quiet{color:#8b93a3;font-size:13px;margin-top:12px}
.gap{background:#fffbe8;border:1px solid #f7e3a8;border-radius:10px;padding:12px 15px;
 color:#7a5b00;font-size:12.5px;margin-top:14px}
.gap b{display:block;margin-bottom:6px;color:#6b4f00}
.gap ul{margin:0;padding-left:18px}
.watch{table-layout:fixed}
.watch th:nth-child(1),.watch td:nth-child(1){width:104px}
.watch th:nth-child(2),.watch td:nth-child(2){width:300px}
.watch th:nth-child(2),.watch th:nth-child(3){text-align:left}
.watch td:first-child{font-weight:600;color:#333b4a}
.watch td:nth-child(2){text-align:left;white-space:normal}
.watch td:nth-child(3){text-align:left;white-space:normal;color:#5d6675;font-size:12.5px}
.foot{color:#9aa1b0;font-size:12px;line-height:1.8;margin-top:8px;padding:0 4px}
.foot b{color:#7c8494}
.disc{background:#f7f8fa;border:1px solid #e8eaee;border-radius:10px;padding:14px 16px;
 color:#7c8494;font-size:12px;margin-top:16px}
"""


def _e(v) -> str:
    return html.escape(str(v)) if v is not None else ""


def _td(v, digits=2, suffix="%"):
    """带涨跌配色的单元格。"""
    if v is None:
        return '<td class="flat">—</td>'
    return f'<td class="{_cls(v)}">{v:+.{digits}f}{suffix}</td>'


def render_html(f: dict) -> str:
    m, mk, se, ev, ho, te, wa = (
        f["meta"], f["market"], f["sentiment"], f["events"],
        f["holdings"], f["technical"], f["watch"],
    )
    regime = mk.get("regime", {})
    P = []
    a = P.append

    a('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width,initial-scale=1">')
    a(f"<title>交易日报 {m['trade_date']}</title><style>{CSS}</style></head><body><div class='wrap'>")

    # 头部
    a("<div class='head'>")
    a(f"<h1>交易日报</h1>")
    a(f"<div class='sub'>{m['trade_date']} 收盘 ｜ 上一交易日 {m.get('prev_date','—')} ｜ "
      f"生成于 {m['generated_at']}</div>")
    a("<div class='badges'>")
    a(f"<span class='badge b-regime'>市场档位 · {_e(regime.get('level','—'))}</span>")
    a(f"<span class='badge b-score'>情绪 {se['score']['score']:.0f} / 100 · "
      f"{_e(se['score']['label'])}</span>")
    if regime.get("position_cap"):
        a(f"<span class='badge b-cap'>建议仓位上限 {regime['position_cap']}%</span>")
    a("</div>")
    a(f"<div class='reason'>{_e(regime.get('reason',''))}</div>")
    sm = f.get("summary", {}).get("lines") or []
    if sm:
        a("<div class='points'>")
        for ln in sm:
            a(f"<div class='pt'>{_e(ln)}</div>")
        a("</div>")
    a("</div>")

    # 1 市场概览
    a("<div class='card'><h2><span class='n'>1</span>市场概览</h2>"
      "<div class='hint'>指数位置与全市场广度，判断今天该用什么姿势</div>")
    a("<table><thead><tr><th>指数</th><th>收盘</th><th>涨跌</th><th>成交额</th>"
      "<th>额环比</th><th>MA20</th><th>MA60</th><th>20日</th></tr></thead><tbody>")
    for i in mk["indices"]:
        a(f"<tr><td>{_e(i['name'])}</td><td>{i['close']:,.2f}</td>{_td(i['pct'])}"
          f"<td>{i['amount_yi']:,.0f} 亿</td>{_td(i['amount_chg'],1)}"
          f"<td>{'上方' if i['above_ma20'] else '下方'}</td>"
          f"<td>{'上方' if i['above_ma60'] else '下方'}</td>{_td(i['ret20'],1)}</tr>")
    a("</tbody></table>")
    b = mk["breadth"]
    a("<div class='kpis'>")
    a(f"<div class='kpi'><div class='k'>涨跌家数</div><div class='v'>"
      f"<span class='up'>{b['up']}</span> / <span class='down'>{b['down']}</span></div>"
      f"<div class='x'>上涨占比 {b['up_ratio']:.1f}%</div></div>")
    a(f"<div class='kpi'><div class='k'>中位数涨跌</div>"
      f"<div class='v {_cls(b['median_pct'])}'>{_sgn(b['median_pct'])}</div>"
      f"<div class='x'>比均值更抗极值</div></div>")
    a(f"<div class='kpi'><div class='k'>涨停 / 跌停</div><div class='v'>"
      f"<span class='up'>{b['limit_up']}</span> / <span class='down'>{b['limit_down']}</span></div>"
      f"<div class='x'>统计口径已区分板块限幅</div></div>")
    a("</div></div>")

    # 2 资金与情绪
    a("<div class='card'><h2><span class='n'>2</span>资金与情绪</h2>"
      "<div class='hint'>钱在不在场，比指数涨跌更能说明问题</div>")
    a("<div class='kpis'>")
    a(f"<div class='kpi'><div class='k'>全市场成交额</div><div class='v'>{se['amount_yi']:,.0f} 亿</div>"
      f"<div class='x'>环比 {_sgn(se['amount_chg'],1)} ｜ 较20日均值 {_sgn(se['amount_vs_ma20'],1)}</div></div>")
    a(f"<div class='kpi'><div class='k'>成交额分位</div><div class='v'>{se['amount_pct_rank60']:.0f}%</div>"
      f"<div class='x'>近 60 个交易日</div></div>")
    a(f"<div class='kpi'><div class='k'>封板率</div>"
      f"<div class='v'>{se['seal_rate']:.0f}%</div>"
      f"<div class='x'>触及涨停 {se['touched_up']} 家</div></div>"
      if se.get("seal_rate") is not None else "")
    a(f"<div class='kpi'><div class='k'>涨5% / 跌5%</div><div class='v'>"
      f"<span class='up'>{se['up5']}</span> / <span class='down'>{se['down5']}</span></div>"
      f"<div class='x'>极端波动家数</div></div>")
    a("</div>")
    sp = se["score"]["parts"]
    a(f"<div style='font-size:13px;color:#5d6675'>情绪分构成：广度 {sp['breadth']:+.1f}、"
      f"涨停 {sp['limit_up']:+.1f}、跌停 {sp['limit_down']:+.1f}、封板 {sp['seal']:+.1f}"
      f"（基准 50 分）</div>")
    a(f"<div class='bar'><i style='width:{se['score']['score']:.0f}%'></i></div>")
    a("<div class='gap'><b>本区块暂缺的数据（不猜，留白）</b><ul>")
    for x in se["missing"]:
        a(f"<li>{_e(x)}</li>")
    a("</ul></div></div>")

    # 3 事件与消息
    a("<div class='card'><h2><span class='n'>3</span>事件与消息</h2>"
      "<div class='hint'>只列可验证事件；无法验证的消息一律不写</div>")
    if not ev["events"]:
        a("<div style='color:#8b93a3'>今日未从行情与财务数据中检测到可验证事件。</div>")
    for e in ev["events"]:
        cls = "ev warn" if e.get("level") == "warn" else "ev"
        a(f"<div class='{cls}'><div class='t'>{_e(e['title'])}</div>")
        if e.get("detail"):
            a(f"<div class='d'>{_e(e['detail'])}</div>")
        if e.get("items"):
            a("<ul>")
            for it in e["items"]:
                a(f"<li><b>{_e(it['name'])}</b>：{_e(it['value'])}</li>")
            a("</ul>")
        a("</div>")
    a("<div class='gap'><b>消息面数据缺口</b><ul>")
    for x in ev["gaps"]:
        a(f"<li>{_e(x)}</li>")
    a("</ul></div></div>")

    # 4 持仓异动
    s = ho.get("summary", {})
    a("<div class='card'><h2><span class='n'>4</span>持仓异动</h2>"
      "<div class='hint'>没触发阈值的票不出现在下面，出现即代表有事</div>")
    if s:
        a("<div class='kpis'>")
        a(f"<div class='kpi'><div class='k'>组合市值</div><div class='v'>{_money(s['mv'])}</div>"
          f"<div class='x'>浮盈 {_money(s['pnl'])}（{_sgn(s['pnl_pct'])}）</div></div>")
        a(f"<div class='kpi'><div class='k'>当日盈亏</div>"
          f"<div class='v {_cls(s['day_ret'])}'>{_money(s['day_pnl'])}</div>"
          f"<div class='x'>组合 {_sgn(s['day_ret'])} ｜ 基准 {_sgn(s['bench_ret'])}</div></div>")
        a(f"<div class='kpi'><div class='k'>超额收益</div>"
          f"<div class='v {_cls(s['excess'])}'>{_sgn(s['excess'])}</div>"
          f"<div class='x'>相对 {_e(s['bench_name'] or '基准')}</div></div>")
        a(f"<div class='kpi'><div class='k'>均线结构</div><div class='v'>"
          f"{s['above_ma20']}/{s['positions']}</div>"
          f"<div class='x'>站上 MA20 ｜ MA60 {s['above_ma60']}/{s['positions']}</div></div>")
        a(f"<div class='kpi'><div class='k'>集中度</div><div class='v'>{s['top3_weight']:.0f}%</div>"
          f"<div class='x'>前三合计 ｜ 最大 {_e(s['max_weight_name'] or '')} {s['max_weight']:.1f}%</div></div>")
        a("</div>")

    if ho["alerts"]:
        for x in ho["alerts"]:
            sev = f" s{x['severity']}" if x.get("severity") else ""
            a(f"<div class='alert{sev}'><div class='h'><div class='nm'>{_e(x['name'])}"
              f"<span class='tag'>{_e(x['ts_code'])}</span>"
              + (f"<span class='tag'>{_e(x.get('tag',''))}</span>" if x.get("tag") else "")
              + f"</div><div>{x['close']:.2f} "
              f"<span class='{_cls(x['day_ret'])}'>{_sgn(x['day_ret'])}</span>"
              + (f" <span style='color:#8b93a3;font-size:12px'>超额 {_sgn(x['excess'])}</span>"
                 if x.get("excess") is not None else "")
              + f" ｜ 浮盈 <span class='{_cls(x['pnl_pct'])}'>{_sgn(x['pnl_pct'])}</span>"
              f" ｜ 权重 {x['weight']:.1f}%</div></div>")
            a("<ul>")
            for r in x["reasons"]:
                a(f"<li>{_e(r)}</li>")
            a("</ul></div>")
    else:
        a("<div style='color:#8b93a3'>今日无持仓触发异动阈值。</div>")

    if ho["quiet"]:
        a("<div class='quiet'>其余 "
          + str(len(ho["quiet"])) + " 只无异常："
          + "、".join(f"{_e(q['name'])}<span class='{_cls(q['day_ret'])}'>"
                      f"({_sgn(q['day_ret'])})</span>" for q in ho["quiet"])
          + "</div>")
    if ho.get("table"):
        a("<div style='margin-top:18px'><div style='font-size:12px;color:#8b93a3;"
          "margin-bottom:8px'>全部持仓（按权重排序）</div>")
        a("<table><thead><tr><th>持仓</th><th>现价</th><th>当日</th><th>权重</th>"
          "<th>浮盈</th><th>MA20</th><th>MA60</th><th>20日位置</th><th>20日涨跌</th>"
          "</tr></thead><tbody>")
        for t in ho["table"]:
            pos_cls = ("up" if (t["pos20"] or 0) >= 0.8
                       else "down" if (t["pos20"] is not None and t["pos20"] <= 0.2)
                       else "flat")
            a(f"<tr><td>{_e(t['name'])}"
              + (f"<span class='tag'>{_e(t['tag'])}</span>" if t.get("tag") else "")
              + f"</td><td>{t['close']:,.2f}</td>{_td(t['day_ret'])}"
              f"<td>{t['weight']:.1f}%</td>{_td(t['pnl_pct'])}"
              f"<td>{_e(t['ma20_side'])}</td><td>{_e(t['ma60_side'])}</td>"
              f"<td class='{pos_cls}'>{_num(t['pos20'])}</td>{_td(t['ret20'],1)}</tr>")
        a("</tbody></table></div>")
    a("</div>")

    # 5 技术面
    a("<div class='card'><h2><span class='n'>5</span>技术面观察</h2>"
      "<div class='hint'>位置比趋势更可靠：同样跌 3%，低位和高位的含义完全不同</div>")
    a("<table><thead><tr><th>标的</th><th>收盘</th><th>MA20</th><th>MA60</th>"
      "<th>20日位置</th><th>ATR</th><th>20日</th><th>60日</th></tr></thead><tbody>")
    for i in te["indices"]:
        a(f"<tr><td>{_e(i['name'])}</td><td>{i['close']:,.2f}</td>"
          f"<td>{_num(i['ma20'])}</td><td>{_num(i['ma60'])}</td><td>—</td><td>—</td>"
          f"{_td(i['ret20'],1)}<td class='flat'>—</td></tr>")
    for h in te["holdings"]:
        pos_cls = "up" if (h["pos20"] or 0) >= 0.8 else "down" if (h["pos20"] or 1) <= 0.2 else "flat"
        a(f"<tr><td>{_e(h['name'])}</td><td>{h['close']:,.2f}</td>"
          f"<td>{_num(h['ma20'])}</td><td>{_num(h['ma60'])}</td>"
          f"<td class='{pos_cls}'>{_num(h['pos20'])}</td>"
          f"<td>{_num(h['atr_pct'],1)}%</td>{_td(h['ret20'],1)}{_td(h['ret60'],1)}</tr>")
    a("</tbody></table>")
    hl = te.get("market", {})
    if hl:
        a(f"<div style='margin-top:12px;color:#5d6675;font-size:13px'>"
          f"全市场创 60 日新高 <b class='up'>{hl['new_high60']}</b> 家、"
          f"新低 <b class='down'>{hl['new_low60']}</b> 家，"
          f"新高占比 {hl['high_ratio']}%"
          f"{'　—— 新高家数偏少，普涨但缺乏领涨主线' if (hl['high_ratio'] or 0) < 5 else ''}</div>")
    a("</div>")

    # 6 明日观察
    a("<div class='card'><h2><span class='n'>6</span>明日观察清单</h2>"
      "<div class='hint'>只给条件框架：若 A 则 B。不给方向预测</div>")
    a("<table class='watch'><thead><tr><th>对象</th><th>若发生</th><th>则</th></tr></thead><tbody>")
    last = None
    for w in wa:
        new_group = w["scope"] != last
        last = w["scope"]
        style = " style='border-top:1px solid #eceef2'" if new_group and last else ""
        a(f"<tr{style}><td>{_e(w['scope']) if new_group else ''}</td>"
          f"<td>{_e(w['cond'])}</td><td>{_e(w['then'])}</td></tr>")
    a("</tbody></table></div>")

    # 尾部
    a("<div class='foot'><b>数据说明</b><br>")
    a(f"生成时间 {m['generated_at']}，行情截至 {m['trade_date']}。"
      f"异动阈值：单日涨跌 {m['thresholds']['ret']}%、量能比 "
      f"{m['thresholds']['vol_ratio']}、ATR 倍数 {m['thresholds']['atr']}。<br>")
    a("涨跌幅已按复权口径修正，除权日不会出现失真的暴跌数字；"
      "持仓收益按原始价与持仓成本计算，与券商账户浮动盈亏口径一致。<br>")
    for g in m["data_gaps"]:
        a(f"⚠️ {_e(g)}<br>")
    a("</div>")
    a(f"<div class='disc'>{_e(DISCLAIMER_MD.replace('> ', '').replace('**',''))}</div>")
    a("</div></body></html>")
    return "".join(P)


def save(f: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    d = Path(out_dir) if out_dir else REPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    tag = f["meta"]["trade_date"].replace("-", "")
    md = d / f"daily_{tag}.md"
    htm = d / f"daily_{tag}.html"
    md.write_text(render_markdown(f), encoding="utf-8")
    htm.write_text(render_html(f), encoding="utf-8")
    return md, htm
