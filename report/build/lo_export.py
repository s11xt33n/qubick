# -*- coding: utf-8 -*-
"""Экспорт docx в PDF через LibreOffice (UNO) с настройкой формул.

Запускается встроенным в LibreOffice интерпретатором:
    "C:\\Program Files\\LibreOffice\\program\\python.exe" lo_export.py <in.docx> <out.pdf>

При импорте формул Word (OMML) LibreOffice ставит им базовый размер 12 pt и свой шрифт, поэтому в PDF
формулы выходят мельче текста. Скрипт открывает документ, задаёт всем формулам 14 pt и Times New Roman
(как у основного текста), обновляет их размеры и экспортирует PDF. Сам docx не меняется.
"""
import os
import subprocess
import sys
import time

import uno
from com.sun.star.beans import PropertyValue

SOFFICE = os.path.join(os.path.dirname(sys.executable), "soffice.exe")
PIPE = "qubik_export_%d" % os.getpid()


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


def kill_stale():
    """Завершить оставшиеся от прошлых запусков экземпляры LibreOffice с нашим профилем (иначе новый зависает)."""
    try:
        out = subprocess.run(["wmic", "process", "where", "name='soffice.bin' or name='soffice.exe'",
                              "get", "ProcessId,CommandLine", "/format:list"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return
    cmd = None
    for line in out.splitlines():
        if line.startswith("CommandLine="):
            cmd = line
        elif line.startswith("ProcessId=") and cmd and "lo_profile_qubik" in cmd:
            subprocess.run(["taskkill", "/F", "/PID", line.split("=", 1)[1].strip()], capture_output=True)
            cmd = None


def main(src, dst):
    kill_stale()
    profile_dir = os.path.join(os.environ.get("TEMP", "."), "lo_profile_qubik")
    try:  # блокировка профиля от принудительно завершённого экземпляра мешает запуску
        os.remove(os.path.join(profile_dir, ".lock"))
    except OSError:
        pass
    profile = uno.systemPathToFileUrl(profile_dir)
    # вывод LibreOffice — в никуда: иначе он держит каналы вызывающего процесса и тот не может завершиться
    proc = subprocess.Popen([SOFFICE, "--headless", "--invisible", "--nologo", "--norestore", "--nodefault",
                             "-env:UserInstallation=" + profile, "--accept=pipe,name=%s;urp;" % PIPE],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        ctx = None
        for _ in range(120):
            try:
                ctx = resolver.resolve("uno:pipe,name=%s;urp;StarOffice.ComponentContext" % PIPE)
                break
            except Exception:
                time.sleep(0.5)
        if ctx is None:
            raise RuntimeError("LibreOffice не запустился")
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(src)), "_blank", 0,
                                           (prop("Hidden", True),))
        objs = doc.getEmbeddedObjects()
        n = 0
        for name in objs.getElementNames():
            o = objs.getByName(name)
            model = o.getEmbeddedObject()
            if model is None or not model.supportsService("com.sun.star.formula.FormulaProperties"):
                continue
            model.BaseFontHeight = 14
            for fp in ("FontNameVariables", "FontNameFunctions", "FontNameNumbers", "FontNameText"):
                setattr(model, fp, "Times New Roman")
            model.FontVariablesIsItalic = True   # переменные — курсивом, функции и числа — прямо
            model.FontFunctionsIsItalic = False
            model.FontNumbersIsItalic = False
            model.FontTextIsItalic = False
            try:
                o.getExtendedControlOverEmbeddedObject().update()
            except Exception:
                pass
            n += 1
        doc.storeToURL(uno.systemPathToFileUrl(os.path.abspath(dst)), (prop("FilterName", "writer_pdf_Export"),))
        print("formulas:", n, flush=True)
        # doc.close() в фоновом режиме может зависнуть — документ не закрываем, процесс завершается ниже
    finally:
        # LibreOffice после terminate() иногда не выходит — завершаем всё дерево процессов сами
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
