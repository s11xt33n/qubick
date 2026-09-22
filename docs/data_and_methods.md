<!-- Материал для отчёта и диссертации: источники данных, протокол, ПО, литература.
     Все ссылки проверены 22.09.2026 (HTTP 200, DOI сверены через Crossref). -->

Прозрачность
# Данные и методы

    Все наборы данных открытые, все цифры на сайте воспроизводимы: ниже — источники данных, полный
      протокол экспериментов, версии программ, сырые результаты для скачивания и список литературы.

## Наборы данных

      Набор | Задача | Объём в экспериментах | Источник | Статья | 

        | Iris | 3 вида ирисов, 4 признака | 150 объектов
          [UCI ML Repository](https://archive.ics.uci.edu/dataset/53/iris)
          [Fisher, 1936](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) |
        | Wine | 3 сорта вина, 13 химических признаков | 178 объектов
          [UCI ML Repository](https://archive.ics.uci.edu/dataset/109/wine) | — |
        | Breast Cancer Wisconsin | доброкачественная / злокачественная опухоль, 30 признаков | 569 объектов
          [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | — |
        | Moons | синтетические «полумесяцы», 2 класса | 400 точек, шум 0.25
          [scikit-learn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) | — |
        | Circles | синтетические концентрические кольца, 2 класса | 400 точек, шум 0.1
          [scikit-learn make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) | — |
        | MNIST | рукописные цифры, 10 классов | 10 000 обучение / 2 000 тест (случайная подвыборка)
          [torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
          [LeCun et al., 1998](https://doi.org/10.1109/5.726791) |
        | Fashion-MNIST | предметы одежды, 10 классов | 10 000 / 2 000
          [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
          [Xiao et al., 2017](https://arxiv.org/abs/1708.07747) |
        | PneumoniaMNIST | рентген грудной клетки: пневмония / норма | 4 708 / 624 (целиком)
          [MedMNIST v2](https://medmnist.com/)
          [Yang et al., 2023](https://arxiv.org/abs/2110.14795); снимки —
            [Kermany et al., 2018](https://doi.org/10.1016/j.cell.2018.02.010) |
        | BreastMNIST | УЗИ молочной железы: доброкачественное / злокачественное | 546 / 156 (целиком)
          [MedMNIST v2](https://medmnist.com/)
          [Yang et al., 2023](https://arxiv.org/abs/2110.14795); снимки —
            [Al-Dhabyani et al., 2020](https://doi.org/10.1016/j.dib.2019.104863) |

    Признаки изображений извлекаются замороженной сетью ResNet18 с весами ImageNet
      ([torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
      [He et al., 2016](https://arxiv.org/abs/1512.03385)): изображение 224×224 → вектор из 512 признаков.
      Все наборы загружаются автоматически официальными загрузчиками библиотек.

## Протокол экспериментов

        - **Разбиение.** Табличные данные — стратифицированно 60 / 20 / 20 % (обучение / валидация / тест). Изображения — фиксированный тест, из обучающей части берётся стратифицированная подвыборка нужного размера (50–2000), 20 % её идёт на валидацию.
        - **Предобработка.** StandardScaler, обученный только на обучающей выборке. Для чистой квантовой модели признаки сжимаются PCA до числа кубитов (тоже только по обучающей выборке) — без утечки информации из теста.
        - **Повторы.** 10 seed на каждую конфигурацию (0–9); seed задаёт и разбиение данных, и инициализацию модели.
        - **Обучение.** Adam, lr = 0.01, батч 32 (64 для 12 кубитов), кросс-энтропия, до 300 эпох (150 для изображений) с ранней остановкой по loss на валидации (терпение 30 / 20 эпох) и возвратом лучших весов.
        - **Метрики.** Accuracy и macro-F1 на тесте; доля предсказываемых классов (выявляет «схлопывание» модели в один класс).
        - **Честное сравнение.** MLP с тем же числом параметров, что у гибрида, и bottleneck-модель — гибрид без квантовой части.
        - **Статистика.** Парный тест Уилкоксона по seed (гибрид против каждой модели), уровень значимости 0.05.

## Программы и оборудование

        - Python 3.12, PyTorch 2.14 (CUDA 12.6), scikit-learn 1.9, NumPy 2.5, pandas 3.0.
        - PennyLane 0.45 — только как эталон для проверки корректности собственного симулятора (158 автотестов).
        - Сервер: 2 × Intel Xeon E5-2698 v4 (80 потоков), 64 ГБ RAM, NVIDIA Tesla V100-SXM2 16 ГБ, Windows 11.
        - Каждая конфигурация эксперимента задана YAML-файлом; идентификатор запуска — хеш всех его параметров.

## Сырые результаты

      Всё, из чего строятся графики, — в открытом виде (средние, разбросы, число запусков, тесты значимости):

        - [tabular.json](data/tabular.json) — табличные данные
        - [vision.json](data/vision.json) — изображения, 4 и 8 кубитов
        - [vision_q12.json](data/vision_q12.json) — изображения, 12 кубитов
        - [sweep.json](data/sweep.json) — кубиты × глубина
        - [ablation.json](data/ablation.json) — абляция схемы
        - [shots.json](data/shots.json) — конечное число измерений
        - [init.json](data/init.json) · [barren_init.json](data/barren_init.json) · [barren.json](data/barren.json) — инициализация и barren plateaus
        - [encoder.json](data/encoder.json) — варианты encoder'а
        - [speed.json](data/speed.json) · [server.json](data/server.json) — скорость и журнал сервера

## Литература

      - Preskill J. Quantum computing in the NISQ era and beyond // Quantum. 2018. Vol. 2. P. 79. [arXiv:1801.00862](https://arxiv.org/abs/1801.00862)
      - Mari A., Bromley T. R., Izaac J., Schuld M., Killoran N. Transfer learning in hybrid classical-quantum neural networks // Quantum. 2020. Vol. 4. P. 340. [arXiv:1912.08278](https://arxiv.org/abs/1912.08278)
      - McClean J. R. et al. Barren plateaus in quantum neural network training landscapes // Nature Communications. 2018. Vol. 9. P. 4812. [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
      - Cerezo M. et al. Cost function dependent barren plateaus in shallow parametrized quantum circuits // Nature Communications. 2021. Vol. 12. P. 1791. [arXiv:2001.00550](https://arxiv.org/abs/2001.00550)
      - Grant E. et al. An initialization strategy for addressing barren plateaus in parametrized quantum circuits // Quantum. 2019. Vol. 3. P. 214. [arXiv:1903.05076](https://arxiv.org/abs/1903.05076)
      - Pérez-Salinas A. et al. Data re-uploading for a universal quantum classifier // Quantum. 2020. Vol. 4. P. 226. [arXiv:1907.02085](https://arxiv.org/abs/1907.02085)
      - Schuld M. et al. Evaluating analytic gradients on quantum hardware // Physical Review A. 2019. Vol. 99. 032331. [arXiv:1811.11184](https://arxiv.org/abs/1811.11184)
      - Bergholm V. et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations. 2018. [arXiv:1811.04968](https://arxiv.org/abs/1811.04968)
      - Paszke A. et al. PyTorch: An imperative style, high-performance deep learning library // NeurIPS. 2019. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
      - He K. et al. Deep residual learning for image recognition // CVPR. 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
      - Kingma D. P., Ba J. Adam: A method for stochastic optimization // ICLR. 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
      - Ioffe S., Szegedy C. Batch normalization // ICML. 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
      - Yang J. et al. MedMNIST v2 — a large-scale lightweight benchmark for 2D and 3D biomedical image classification // Scientific Data. 2023. [arXiv:2110.14795](https://arxiv.org/abs/2110.14795)
      - Kermany D. S. et al. Identifying medical diagnoses and treatable diseases by image-based deep learning // Cell. 2018. Vol. 172, № 5. [doi:10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
      - Al-Dhabyani W. et al. Dataset of breast ultrasound images // Data in Brief. 2020. Vol. 28. 104863. [doi:10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
      - LeCun Y. et al. Gradient-based learning applied to document recognition // Proceedings of the IEEE. 1998. Vol. 86, № 11. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
      - Xiao H., Rasul K., Vollgraf R. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. 2017. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)
      - Fisher R. A. The use of multiple measurements in taxonomic problems // Annals of Eugenics. 1936. Vol. 7, № 2. [doi:10.1111/j.1469-1809.1936.tb02137.x](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)
      - Wilcoxon F. Individual comparisons by ranking methods // Biometrics Bulletin. 1945. Vol. 1, № 6. P. 80–83.
