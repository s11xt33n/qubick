# -*- coding: utf-8 -*-
"""Сборка отчёта: шаблон — прошлый отчёт (титул, стили, таблицы, подписи), содержание — report_content.py.

    python build_report.py <root> [toc.json]
"""
import ast, copy, json, pathlib, sys, textwrap
import docx
from docx.shared import Cm
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8")
root = pathlib.Path(sys.argv[1])
toc_pages = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")) if len(sys.argv) > 2 else {}
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import report_content_sem2 as C

SRC = root / "src" / "Отчет студента-09-535-2 сем.docx"
OUT = root / "report" / "Отчет_Филадельфов_09-535_2_сем.docx"
FIG = root / "report" / "sem2" / "fig"

D = docx.Document(SRC)
body = D.element.body
kids = list(body.iterchildren())
T = {name: copy.deepcopy(kids[i]) for name, i in {
    "pb": 40, "h1": 41, "p": 42, "h2": 48, "tcap": 58, "tbl2": 59, "fig": 61, "fcap": 62,
    "tbl3": 76, "tbl4": 69, "tbl7": 92, "h1c": 107, "tblcomp": 118, "ref": 122,
    "apx": 135, "atitle": 136, "aauth": 137, "asup": 138, "aaff": 139, "amail": 140, "ahead": 141,
    "ap": 142, "ali": 152, "atcap": 170, "atbl": 177, "aref": 184, "apx2": 193, "code": 195}.items()}
sdt = kids[39]
sect = kids[-1]
assert sect.tag == qn("w:sectPr")
# удалить всё после содержания (и висячие закладки перед ним)
for el in kids[34:39] + kids[40:-1]:
    body.remove(el)


def W(tag):
    return qn("w:" + tag)


def set_text(el, text, bold=None):
    """Оставить первый текстовый run абзаца с его форматированием и записать в него текст."""
    runs = [r for r in el.iter(W("r")) if r.find(W("t")) is not None] or list(el.iter(W("r")))
    if not runs:
        r = copy.deepcopy(T["p"].find(W("r"))); el.append(r); runs = [r]
    keep = runs[0]
    for child in list(el):
        if child.tag not in (W("pPr"),) and child is not keep and not (keep in list(child.iter())):
            el.remove(child)
    if keep.getparent() is not el:  # run был внутри гиперссылки/поля — вынести
        el.append(keep)
    for c in list(keep):
        if c.tag != W("rPr"):
            keep.remove(c)
    t = keep.makeelement(W("t"), {}); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); keep.append(t)
    if bold is not None:
        rpr = keep.find(W("rPr"))
        if rpr is None:
            rpr = keep.makeelement(W("rPr"), {}); keep.insert(0, rpr)
        for b in rpr.findall(W("b")) + rpr.findall(W("bCs")):
            rpr.remove(b)
        if bold:
            rpr.append(rpr.makeelement(W("b"), {}))
    return el


def keep_next(el):
    ppr = el.find(W("pPr"))
    if ppr is None:
        ppr = el.makeelement(W("pPr"), {}); el.insert(0, ppr)
    if ppr.find(W("keepNext")) is None:
        ppr.insert(1 if ppr.find(W("pStyle")) is not None else 0, ppr.makeelement(W("keepNext"), {}))
    return el


def add(el):
    sect.addprevious(el)
    return el


def para(kind, text, bold=None):
    return add(set_text(copy.deepcopy(T[kind]), text, bold))


def pagebreak():
    return add(copy.deepcopy(T["pb"]))


def figure(fname, width_cm, caption):
    el = copy.deepcopy(T["fig"])
    for r in list(el.iter(W("r"))):
        r.getparent().remove(r)
    ppr = el.find(W("pPr"))
    for bad in ppr.findall(W("ind")) + ppr.findall(W("mirrorIndents")):
        ppr.remove(bad)
    # явный нулевой отступ: иначе отступ первой строки из стиля сдвигает рисунок вправо
    ind = ppr.makeelement(W("ind"), {W("left"): "0", W("right"): "0", W("firstLine"): "0"})
    jc = ppr.find(W("jc"))
    (jc.addprevious(ind) if jc is not None else ppr.append(ind))
    add(keep_next(el))
    p = docx.text.paragraph.Paragraph(el, D._body)
    p.add_run().add_picture(str(FIG / fname), width=Cm(width_cm))
    cap = para("fcap", caption)
    cppr = cap.find(W("pPr"))
    for bad in cppr.findall(W("ind")) + cppr.findall(W("mirrorIndents")):
        cppr.remove(bad)
    cind = cppr.makeelement(W("ind"), {W("left"): "0", W("right"): "0", W("firstLine"): "0"})
    cjc = cppr.find(W("jc"))
    (cjc.addprevious(cind) if cjc is not None else cppr.append(cind))


def table(tpl, header, rows, widths_cm, tcap_kind="tcap", caption=None, align=None):
    if caption:
        keep_next(para(tcap_kind, caption))
    tbl = copy.deepcopy(T[tpl])
    trs = tbl.findall(W("tr"))
    proto_h, proto_d = trs[0], trs[1]
    for tr in trs:
        tbl.remove(tr)
    n = len(header)
    total = sum(widths_cm)
    dxa = [int(w * 567) for w in widths_cm]
    grid = tbl.find(W("tblGrid"))
    for g in list(grid):
        grid.remove(g)
    for w in dxa:
        grid.append(grid.makeelement(W("gridCol"), {W("w"): str(w)}))
    tblpr = tbl.find(W("tblPr"))
    tw = tblpr.find(W("tblW")) if tblpr is not None else None
    if tw is not None:
        tw.set(W("w"), str(sum(dxa))); tw.set(W("type"), "dxa")
    align = (align or "l" * n).ljust(n, "l")
    mar = tblpr.find(W("tblCellMar"))
    if tpl != "atbl":
        if mar is None:
            mar = tblpr.makeelement(W("tblCellMar"), {}); tblpr.find(W("tblLook")).addprevious(mar)
        for c in list(mar):
            mar.remove(c)
        for side in ("left", "right"):
            mar.append(mar.makeelement(W(side), {W("w"): "85", W("type"): "dxa"}))

    def fix_para(p, jc):
        ppr = p.find(W("pPr"))
        if ppr is None:
            ppr = p.makeelement(W("pPr"), {}); p.insert(0, ppr)
        for bad in ppr.findall(W("ind")) + ppr.findall(W("mirrorIndents")) + ppr.findall(W("jc")) + ppr.findall(W("spacing")):
            ppr.remove(bad)
        rpr = ppr.find(W("rPr"))
        sp = ppr.makeelement(W("spacing"), {W("before"): "0", W("after"): "0", W("line"): "240", W("lineRule"): "auto"})
        j = ppr.makeelement(W("jc"), {W("val"): {"l": "left", "c": "center", "r": "right"}[jc]})
        if rpr is not None:
            rpr.addprevious(sp); rpr.addprevious(j)
        else:
            ppr.append(sp); ppr.append(j)

    def make_row(proto, values, is_head=False):
        tr = copy.deepcopy(proto)
        trpr = tr.find(W("trPr"))
        if trpr is None:
            trpr = tr.makeelement(W("trPr"), {}); tr.insert(0, trpr)
        for bad in trpr.findall(W("trHeight")) + trpr.findall(W("tblHeader")) + trpr.findall(W("cantSplit")):
            trpr.remove(bad)
        trpr.insert(0, trpr.makeelement(W("cantSplit"), {}))
        if is_head:
            trpr.append(trpr.makeelement(W("tblHeader"), {}))
        tcs = tr.findall(W("tc"))
        while len(tcs) < n:
            tcs[-1].addnext(copy.deepcopy(tcs[-1])); tcs = tr.findall(W("tc"))
        for extra in tcs[n:]:
            tr.remove(extra)
        tcs = tr.findall(W("tc"))
        for tc, v, w, a in zip(tcs, values, dxa, align):
            tcpr = tc.find(W("tcPr"))
            if tcpr is not None:
                for bad in tcpr.findall(W("gridSpan")) + tcpr.findall(W("vMerge")):
                    tcpr.remove(bad)
                tcw = tcpr.find(W("tcW"))
                if tcw is not None:
                    tcw.set(W("w"), str(w)); tcw.set(W("type"), "dxa")
            ps = tc.findall(W("p"))
            for extra in ps[1:]:
                tc.remove(extra)
            set_text(ps[0], v)
            fix_para(ps[0], "c" if is_head else a)
            if tcpr is not None and tcpr.find(W("noWrap")) is not None:
                tcpr.remove(tcpr.find(W("noWrap")))
        tbl.append(tr)
    make_row(proto_h, header, True)
    for r in rows:
        make_row(proto_d, r)
    add(tbl)
    return tbl


def source_of(path, names):
    src = (root / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out.append(ast.get_source_segment(src, node))
    return "\n\n".join(out)


def code_block(text):
    for line in text.splitlines():
        lead = len(line) - len(line.lstrip(" "))
        el = para("code", " " * lead + line.lstrip(" ") if line.strip() else " ")
        el.find(W("pPr")).find(W("spacing")).set(W("after"), "0")
        for sz in el.iter(W("sz"), W("szCs")):
            sz.set(W("val"), "20")


# ------------------------------------------------------------------ содержание (TOC)
content = sdt.find(W("sdtContent"))
tps = content.findall(W("p"))
entry_ppr = copy.deepcopy(tps[1].find(W("pPr")))
text_run = next(r for r in tps[1].iter(W("r")) if r.find(W("t")) is not None and (r.find(W("t")).text or "").strip())
entry_rpr = copy.deepcopy(text_run.find(W("rPr")))
for p in tps[1:]:
    content.remove(p)
for title, key in C.TOC:
    p = content.makeelement(W("p"), {}); p.append(copy.deepcopy(entry_ppr))
    for kind, val in (("t", title), ("tab", None), ("t", str(toc_pages.get(key, "")))):
        r = p.makeelement(W("r"), {})
        if entry_rpr is not None:
            r.append(copy.deepcopy(entry_rpr))
        if kind == "tab":
            r.append(r.makeelement(W("tab"), {}))
        else:
            t = r.makeelement(W("t"), {}); t.text = val
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(t)
        p.append(r)
    content.append(p)

# ------------------------------------------------------------------ текст по блокам
def break_before(el):
    """Раздел с новой страницы: свойство абзаца, а не пустой абзац с разрывом (он давал лишние пустые страницы)."""
    ppr = el.find(W("pPr"))
    if ppr is None:
        ppr = el.makeelement(W("pPr"), {}); el.insert(0, ppr)
    ppr.insert(1 if ppr.find(W("pStyle")) is not None else 0, ppr.makeelement(W("pageBreakBefore"), {}))


new_page = False
for b in C.BLOCKS:
    kind = b[0]
    if kind == "pb":
        new_page = True
        continue
    n_before = len(body)
    if kind in ("h1", "h1c", "apx", "apx2"):
        para(kind, b[1])
    elif kind == "ch":
        keep_next(para("h2", b[1]))
    elif kind == "sub":
        keep_next(para("p", b[1], bold=True))
    elif kind == "p":
        para("p", b[1])
    elif kind == "table":
        _, tpl, cap, head, rows, widths, align = b
        table(tpl, head, rows, widths, caption=cap, align=align)
    elif kind == "fig":
        figure(b[1], b[2], b[3])
    elif kind == "refs":
        for r in b[1]:
            para("ref", r)
    elif kind == "code":
        code_block(b[1])
    elif kind == "art":  # статья в формате шаблона: заголовок, автор, руководитель, организация, почта, текст
        _, title, author, sup, aff, mail, items, refs = b
        for k, v in (("atitle", title), ("aauth", author), ("asup", sup), ("aaff", aff), ("amail", mail)):
            para(k, v)
        for it in items:
            if it[0] == "h":
                keep_next(para("ahead", it[1]))
            elif it[0] == "p":
                para("ap", it[1])
            elif it[0] == "li":
                para("ali", it[1])
            elif it[0] == "tcap":
                keep_next(para("atcap", it[1]))
            elif it[0] == "table":
                table("atbl", it[1], it[2], [5.0, 3.8, 3.8, 3.9], align="lccc")
        keep_next(para("ahead", "ЛИТЕРАТУРА"))
        for r in refs:
            para("aref", r)
    else:
        raise ValueError(kind)
    if new_page:  # первый добавленный элемент блока начинает новую страницу
        first = list(body)[n_before - 1]
        if first.tag == W("p"):
            break_before(first)
        else:
            pb = copy.deepcopy(T["pb"]); first.addprevious(pb)
        new_page = False

# номера страниц — Times New Roman 12 (в шаблоне поле PAGE без шрифта и берёт шрифт по умолчанию)
for sec in D.sections:
    for ft in (sec.footer, sec.first_page_footer, sec.even_page_footer):
        for el in ft._element.iter(W("r"), W("pPr")):
            target = el if el.tag == W("r") else el
            rpr = target.find(W("rPr"))
            if rpr is None:
                rpr = target.makeelement(W("rPr"), {})
                target.insert(0, rpr) if el.tag == W("r") else target.append(rpr)
            for old_el in rpr.findall(W("rFonts")) + rpr.findall(W("sz")) + rpr.findall(W("szCs")):
                rpr.remove(old_el)
            rpr.insert(0, rpr.makeelement(W("rFonts"), {W(a): "Times New Roman" for a in ("ascii", "hAnsi", "cs", "eastAsia")}))
            rpr.append(rpr.makeelement(W("sz"), {W("val"): "24"}))
            rpr.append(rpr.makeelement(W("szCs"), {W("val"): "24"}))

OUT.parent.mkdir(exist_ok=True)
D.save(OUT)
print("saved", OUT)
