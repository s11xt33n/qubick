# -*- coding: utf-8 -*-
"""Индивидуальное задание: ровные линии для заполняемых полей вместо «____».

    python fix_assignment.py <in.docx> <out.docx>

Подчёркивания из символов «_» стоят ниже линии подчёркивания текста и дают «ступеньку». Здесь поле —
это подчёркнутый текст плюс подчёркнутая табуляция до нужной позиции: линия получается сплошной,
на одной высоте, а подписи под полями «(подпись)», «(ФИО)» выравниваются по центру своих полей.
"""
import copy
import re
import sys

import docx
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8")
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def W(tag):
    return qn("w:" + tag)


def text_of(el):
    return "".join(t.text or "" for t in el.iter(W("t")))


def is_u(run):
    rpr = run.find(W("rPr"))
    u = rpr.find(W("u")) if rpr is not None else None
    return u is not None and u.get(W("val")) not in (None, "none")


def base_rpr(par, underlined):
    """rPr первого run с текстом нужного вида (подчёркнутый или нет)."""
    for r in par.iter(W("r")):
        if r.find(W("t")) is not None and (text_of(r).strip()) and is_u(r) == underlined:
            return copy.deepcopy(r.find(W("rPr")))
    for r in par.iter(W("r")):
        if r.find(W("rPr")) is not None:
            rpr = copy.deepcopy(r.find(W("rPr")))
            for u in rpr.findall(W("u")):
                rpr.remove(u)
            if underlined:
                rpr.append(rpr.makeelement(W("u"), {W("val"): "single"}))
            return rpr
    return None


def run(par, rpr, text=None, tab=False, underline=False, br=False):
    r = par.makeelement(W("r"), {})
    rp = copy.deepcopy(rpr) if rpr is not None else r.makeelement(W("rPr"), {})
    for u in rp.findall(W("u")):
        rp.remove(u)
    if underline:
        rp.append(rp.makeelement(W("u"), {W("val"): "single"}))
    r.append(rp)
    if br:
        r.append(r.makeelement(W("br"), {}))
    if tab:
        r.append(r.makeelement(W("tab"), {}))
    if text:
        t = r.makeelement(W("t"), {})
        t.text = text
        t.set(XML_SPACE, "preserve")
        r.append(t)
    par.append(r)
    return r


def set_tabs(par, stops):
    ppr = par.find(W("pPr"))
    if ppr is None:
        ppr = par.makeelement(W("pPr"), {})
        par.insert(0, ppr)
    for old in ppr.findall(W("tabs")):
        ppr.remove(old)
    tabs = ppr.makeelement(W("tabs"), {})
    for kind, pos in stops:
        tabs.append(tabs.makeelement(W("tab"), {W("val"): kind, W("pos"): str(pos)}))
    anchor = next((ppr.find(W(t)) for t in ("spacing", "ind", "jc", "rPr") if ppr.find(W(t)) is not None), None)
    (anchor.addprevious(tabs) if anchor is not None else ppr.append(tabs))


def clear_runs(par, keep_first=0):
    """Удалить из абзаца всё, кроме pPr и первых keep_first run'ов с текстом."""
    kept = 0
    for child in list(par):
        if child.tag == W("pPr"):
            continue
        if child.tag == W("r") and kept < keep_first:
            kept += 1
            continue
        par.remove(child)


def find(body, pattern):
    return next(p for p in body.iter(W("p")) if re.search(pattern, text_of(p)))


WS = "  	"


def _plain_copy(r, text):
    new = copy.deepcopy(r)
    rpr = new.find(W("rPr"))
    if rpr is not None:
        for u in rpr.findall(W("u")):
            rpr.remove(u)
    for c in list(new):
        if c.tag != W("rPr"):
            new.remove(c)
    t = new.makeelement(W("t"), {})
    t.text = text
    t.set(XML_SPACE, "preserve")
    new.append(t)
    return new


def tidy_underlines(par):
    """Пробелы по краям подчёркнутого фрагмента (серии подчёркнутых run подряд) — в обычный текст,
    чтобы линия не торчала за словом; пробелы внутри серии остаются подчёркнутыми."""
    runs = [r for r in par.iter(W("r")) if r.find(W("t")) is not None or r.find(W("tab")) is not None]
    i = 0
    while i < len(runs):
        if not is_u(runs[i]) or runs[i].find(W("t")) is None:
            i += 1
            continue
        j = i
        while j + 1 < len(runs) and is_u(runs[j + 1]):
            j += 1
        first, last = runs[i], runs[j]
        if last.find(W("t")) is not None:
            t = last.find(W("t"))
            core = (t.text or "").rstrip(WS)
            tail = (t.text or "")[len(core):]
            if tail and core:
                t.text = core
                t.set(XML_SPACE, "preserve")
                last.addnext(_plain_copy(last, " "))
        t = first.find(W("t"))
        core = (t.text or "").lstrip(WS)
        lead = (t.text or "")[:len(t.text or "") - len(core)]
        if lead and core:
            t.text = core
            t.set(XML_SPACE, "preserve")
            first.addprevious(_plain_copy(first, " "))
        i = j + 1


def main(src, dst):
    D = docx.Document(src)
    body = D.element.body
    sp = body.find(W("sectPr"))
    pg, mar = sp.find(W("pgSz")), sp.find(W("pgMar"))
    width = int(pg.get(W("w"))) - int(mar.get(W("left"))) - int(mar.get(W("right")))  # ширина текста, dxa

    # --- «метка: подчёркнутое значение ____» -> значение + подчёркнутая табуляция до правого поля
    for pattern in (r"^Место прохождения", r"^Обучающийся", r"^Руководитель практики от Университета\s*_"):
        par = find(body, pattern)
        for r in list(par.iter(W("r"))):
            t = r.find(W("t"))
            if t is not None and t.text and set(t.text.strip()) <= {"_"} and t.text.strip():
                r.getparent().remove(r)
            elif t is not None and t.text and "_" in t.text and not is_u(r):
                t.text = t.text.replace("_", "").rstrip() + " "
                t.set(XML_SPACE, "preserve")
        tidy_underlines(par)
        set_tabs(par, [("right", width)])
        run(par, base_rpr(par, True), tab=True, underline=True)

    # подписи под этими полями — по центру подчёркнутой части строки
    def center_caption(pattern, center):
        par = find(body, pattern)
        rpr = base_rpr(par, False)
        caption = text_of(par).strip()
        clear_runs(par)
        set_tabs(par, [("center", center)])
        jc = par.find(W("pPr")).find(W("jc"))
        if jc is not None:
            jc.set(W("val"), "left")
        run(par, rpr, tab=True)
        run(par, rpr, caption)

    center_caption(r"^\s*\(наименование организации", width // 2)
    center_caption(r"^\s*\(ФИО, курс, группа\)", 5790)
    center_caption(r"^\s*\(ФИО, должность, ученое звание\)", 7270)

    # --- подписи руководителей: [должность] [подпись] [ФИО] — три поля одной линией
    f1, f2a, f2b, f3a = 4200, 4700, 6900, 7300
    c1, c2, c3 = f1 // 2, (f2a + f2b) // 2, (f3a + width) // 2
    for who in (r"Андрианова А\.А\.$", r"Васильев А\. В\.$"):
        par = next(p for p in body.iter(W("p")) if re.search(who, text_of(p).strip()) and "_" in text_of(p))
        u_runs = [r for r in par.iter(W("r")) if is_u(r) and text_of(r).strip()]
        position = " ".join(text_of(r).strip() for r in u_runs[:-1]).strip()
        name = text_of(u_runs[-1]).strip()
        rpr = base_rpr(par, True)
        clear_runs(par)
        set_tabs(par, [("center", c1), ("left", f1), ("left", f2a), ("left", f2b), ("left", f3a),
                       ("center", c3), ("right", width)])
        run(par, rpr, tab=True, underline=True)
        run(par, rpr, position, underline=True)
        run(par, rpr, tab=True, underline=True)
        run(par, rpr, tab=True)
        run(par, rpr, tab=True, underline=True)
        run(par, rpr, tab=True)
        run(par, rpr, tab=True, underline=True)
        run(par, rpr, name, underline=True)
        run(par, rpr, tab=True, underline=True)
        cap = par.getnext()
        while cap is not None and not text_of(cap).strip():
            cap = cap.getnext()
        crpr = base_rpr(cap, False)
        clear_runs(cap)
        set_tabs(cap, [("center", c1), ("center", c2), ("center", c3)])
        for piece in ("(должность, ученое звание)", "(подпись)", "(ФИО)"):
            run(cap, crpr, tab=True)
            run(cap, crpr, piece)

    # --- «ОЗНАКОМЛЕН(А) ____ Филадельфов Т. Г. дата»: с новой строки, подпись и ФИО — полями
    par = find(body, r"ОЗНАКОМЛЕН")
    runs = list(par.iter(W("r")))
    k = next(i for i, r in enumerate(runs) if "ОЗНАКОМЛЕН" in text_of(r))
    tail = runs[k + 1:]
    name = " ".join(text_of(r).strip() for r in tail if is_u(r) and text_of(r).strip())
    date = next(m.group(0) for m in [re.search(r"\d{2}\.\d{2}\.\d{4}", text_of(par))] if m)
    urpr = base_rpr(par, True)
    prpr = copy.deepcopy(runs[k - 1].find(W("rPr"))) if k else base_rpr(par, False)
    for r in tail:
        r.getparent().remove(r)
    for x in list(par):
        if x.tag in (W("proofErr"), W("bookmarkStart"), W("bookmarkEnd")):
            par.remove(x)
    a1, a2, b1, b2, d0 = 700, 3300, 3700, 6500, 7000
    set_tabs(par, [("left", a1), ("left", a2), ("left", b1), ("center", (b1 + b2) // 2), ("left", b2), ("left", d0)])
    run(par, urpr, br=True)
    run(par, prpr, tab=True)
    run(par, urpr, tab=True, underline=True)
    run(par, prpr, tab=True)
    run(par, urpr, tab=True, underline=True)
    run(par, urpr, name, underline=True)
    run(par, urpr, tab=True, underline=True)
    run(par, prpr, tab=True)
    run(par, prpr, date)
    cap = par.getnext()
    while cap is not None and not text_of(cap).strip():
        cap = cap.getnext()
    crpr = base_rpr(cap, False)
    clear_runs(cap)
    set_tabs(cap, [("center", (a1 + a2) // 2), ("center", (b1 + b2) // 2)])
    for piece in ("(подпись)", "(ФИО обучающегося)"):
        run(cap, crpr, tab=True)
        run(cap, crpr, piece)

    # --- сроки в таблице: «с ДД.ММ по ДД.ММ», подчёркнуты только даты (скрытый белый «%» шаблона сохраняется)
    for tc in body.iter(W("tc")):
        par = tc.find(W("p"))
        if par is None:
            continue
        m = re.fullmatch(r"[\s ]*с[\s ]*(\d\d\.\d\d)[\s ]*по[\s ]*(\d\d\.\d\d)[\s %]*", text_of(par))
        if not m:
            continue
        runs = [r for r in par.iter(W("r"))]
        hidden = [r for r in runs if r.find(W("rPr")) is not None and r.find(W("rPr")).find(W("color")) is not None
                  and r.find(W("rPr")).find(W("color")).get(W("val")) == "FFFFFF"]
        urpr, prpr = base_rpr(par, True), base_rpr(par, False)
        for r in runs:
            if r not in hidden:
                r.getparent().remove(r)
        for x in list(par):
            if x.tag == W("proofErr"):
                par.remove(x)
        anchor = hidden[0] if hidden else None
        new_runs = []
        for text, u in (("с ", False), (m.group(1), True), (" по ", False), (m.group(2), True)):
            new_runs.append(run(par, urpr if u else prpr, text, underline=u))
        if anchor is not None:  # новые run — перед скрытым маркером
            for r in new_runs:
                anchor.addprevious(r)

    # --- прочие подчёркнутые значения: без хвостов подчёркивания
    for par in body.iter(W("p")):
        tidy_underlines(par)
        for r in par.iter(W("r")):
            t = r.find(W("t"))
            if t is not None and t.text and not is_u(r) and re.fullmatch(r"[\s ]*по[\s ]*", t.text):
                t.text = " по "
                t.set(XML_SPACE, "preserve")
    D.save(dst)
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
