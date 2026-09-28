# -*- coding: utf-8 -*-
"""Индивидуальное задание на 3 семестр.

Бланк 3 семестра пришёл только в PDF, поэтому за основу взят docx 2 семестра — это тот же
бланк КФУ, отличаются вид практики, учебный год, курс, сроки и строки таблицы.

    python sem3_assignment.py <root>
"""
import copy
import pathlib
import sys

import docx
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8")
root = pathlib.Path(sys.argv[1])
SRC = root / "src" / "Филадельфов_Т.Г._инд.задание 09-535- 2 сем (МО и КЗ).docx"
OUT = root / "report" / "sem3" / "Индивидуальное_задание_Филадельфов_09-535_3_сем.docx"

TASKS = [
    ("Исследование методов построения и обучения гибридных квантово-классических нейронных сетей "
     "и проблемы затухания градиентов.", "01.09", "30.09", "Отчет по практике"),
    ("Разработка библиотеки для гибридных нейронных сетей и методики их сравнения с классическими "
     "и квантовыми моделями.", "15.09", "31.10", "Отчет по практике"),
    ("Программная реализация библиотеки, проведение вычислительных экспериментов, в том числе "
     "на медицинских изображениях, анализ результатов.", "01.10", "25.12", "Отчет по практике"),
    ("Выступление с докладом на научном семинаре кафедры", "01.11", "25.12", "Выступление с докладом"),
    ("Подготовка текста публикации по результатам исследований", "01.11", "25.12", "Рукопись публикации"),
    ("Оформление отчета по практике", "26.12", "30.12", "Отчет по практике"),
]


def texts(el):
    return [t for t in el.iter(qn("w:t"))]


def replace_run(par, old, new):
    for t in texts(par):
        if t.text == old:
            t.text = new
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            return
    raise KeyError(old)


def set_cell(tc, text):
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    p = ps[0]
    runs = [r for r in p.findall(qn("w:r")) if r.find(qn("w:t")) is not None]
    for r in runs[1:]:
        p.remove(r)
    for el in list(p):  # маркеры списка, закладки и пр.
        if el.tag not in (qn("w:pPr"), qn("w:r")):
            p.remove(el)
    t = runs[0].find(qn("w:t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def set_dates(tc, a, b):
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    ts = texts(ps[0])  # «с », дата, « по », дата, «%»
    ts[1].text, ts[3].text = a, b


d = docx.Document(SRC)
body = list(d.element.body.iterchildren())
replace_run(body[8], "учебную (научно-исследовательская работа (получение первичных навыков "
                     "научно-исследовательской работы)) ", "производственную (технологическую (проектно-технологическую))")
replace_run(body[10], "(2025/2026 учебный год)", "(2026/2027 учебный год)")
replace_run(body[17], ", 1 курс,  группа 09-535", ", 2 курс, группа 09-535")
replace_run(body[19], "с 09.02.2026 – 06.06.2026  ", "01.09.2026 – 30.12.2026")
replace_run(body[37], "      09.02.2026", "      01.09.2026")

tbl = body[25]
rows = tbl.findall(qn("w:tr"))
proto = copy.deepcopy(rows[5])  # простая строка без списка
for r in rows[4:]:
    tbl.remove(r)
for k in range(3, len(TASKS)):
    tbl.append(copy.deepcopy(proto))
rows = tbl.findall(qn("w:tr"))[1:]
assert len(rows) == len(TASKS)
for i, (tr, (task, a, b, form)) in enumerate(zip(rows, TASKS), 1):
    num, cell, dates, frm = tr.findall(qn("w:tc"))
    set_cell(num, f"{i}.")
    set_cell(cell, task)
    set_dates(dates, a, b)
    set_cell(frm, form)
    ppr = cell.find(qn("w:p")).find(qn("w:pPr"))
    if ppr is not None and ppr.find(qn("w:jc")) is not None:
        ppr.find(qn("w:jc")).set(qn("w:val"), "left")

# ширины столбцов как в бланке 3 семестра: №, задание, сроки, форма (см)
WIDTHS = [int(cm * 567) for cm in (1.0, 9.7, 4.2, 3.0)]
grid = tbl.find(qn("w:tblGrid"))
for g, w in zip(grid.findall(qn("w:gridCol")), WIDTHS):
    g.set(qn("w:w"), str(w))
for tr in tbl.findall(qn("w:tr")):
    for tc, w in zip(tr.findall(qn("w:tc")), WIDTHS):
        tcw = tc.find(qn("w:tcPr")).find(qn("w:tcW"))
        tcw.set(qn("w:w"), str(w)); tcw.set(qn("w:type"), "dxa")
    trpr = tr.find(qn("w:trPr"))
    if trpr is None:
        trpr = tr.makeelement(qn("w:trPr"), {}); tr.insert(0, trpr)
    if trpr.find(qn("w:cantSplit")) is None:
        trpr.insert(0, trpr.makeelement(qn("w:cantSplit"), {}))

OUT.parent.mkdir(parents=True, exist_ok=True)
d.save(OUT)
print("saved", OUT)
