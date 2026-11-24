"""
Лабораторная работа № 9. Вариант 13.
Дисциплина: Теория алгоритмов.

Способы представления графов и алгоритмы обхода в ширину и в глубину.

Исполнитель: Кугушева Арина, группа БИ2-1.

Условия варианта 13 (таблицы 1–3 методических указаний):
    тип графа ............. Ц — цепочка (линейный граф), неориентированный
    размеры n ............. 5000; 50 000; 100 000
    задача варианта ....... глубина обхода и память
    дополнительное ........ И6: график зависимости времени обхода от плотности графа
                            при фиксированном n (1 %, 5 %, 20 %, 50 %, 100 % рёбер от максимума),
                            поиск точки, где матрица смежности становится выгоднее списков

Запуск:
    python lab9_variant13.py                 # проверка, замеры и таблицы в консоль
    python lab9_variant13.py --plots plots   # дополнительно сохранить графики (нужен matplotlib)
    python lab9_variant13.py --json out.json # сохранить результаты в JSON

Требуется Python 3.10+. Сторонние библиотеки нужны только для графиков.
Матрица смежности для n = 50 000 и 100 000 не строится (потребовалась бы память порядка
гигабайт и десятков гигабайт): для этих размеров приводятся расчётные оценки.
"""
import argparse
import gc
import itertools
import json
import random
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from collections import deque

SIZES = [5000, 50_000, 100_000]                      # n из варианта 13
EXT_SIZES = [5000, 10_000, 20_000, 50_000, 100_000]  # ряд для графика времени списков
MATRIX_SIZES = [500, 1000, 2000, 5000]               # размеры, где матрица строится реально
DENSITY_N = 1000                                     # фиксированное n в задании И6
DENSITIES = [0.01, 0.05, 0.20, 0.50, 1.00]           # доли рёбер от максимума
HAS_EDGE_QUERIES = 20_000                            # запросов «есть ли ребро» в замере И6
SEED = 42
REPEATS = 5


# --------------------------------------------------------------------------
# Представления графа (одинаковый интерфейс: add_edge, has_edge, neighbors)
# --------------------------------------------------------------------------
class MatrixGraph:
    """Неориентированный граф на матрице смежности (список списков 0/1). Память Θ(n²)."""

    def __init__(self, n):
        self.n = n
        self.m = 0
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1
        self.matrix[v][u] = 1
        self.m += 1

    def has_edge(self, u, v):
        return self.matrix[u][v] == 1

    def neighbors(self, v):
        row = self.matrix[v]
        return [u for u in range(self.n) if row[u]]


class MatrixGraphBytes(MatrixGraph):
    """Матрица смежности, в которой строка — bytearray (1 байт на ячейку).
    Нужна для сравнения расхода памяти: в Python список 0/1 хранит 8-байтовую ссылку на ячейку."""

    def __init__(self, n):
        self.n = n
        self.m = 0
        self.matrix = [bytearray(n) for _ in range(n)]


class ListGraph:
    """Неориентированный граф на списках смежности. Память Θ(n + m)."""

    def __init__(self, n):
        self.n = n
        self.m = 0
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.m += 1

    def has_edge(self, u, v):
        return v in self.adj[u]

    def neighbors(self, v):
        return self.adj[v]


# --------------------------------------------------------------------------
# Генераторы графов
# --------------------------------------------------------------------------
def make_chain(cls, n):
    """Цепочка 0 — 1 — 2 — … — (n−1): m = n − 1, диаметр n − 1."""
    ids = list(range(n))                      # номера вершин — общие объекты, как в реальном графе
    g = cls(n)
    for i in range(n - 1):
        g.add_edge(ids[i], ids[i + 1])
    return g


def random_edges(n, density, seed=SEED):
    """Случайный набор рёбер: доля density от максимума n(n−1)/2 (без петель и повторов)."""
    rng = random.Random(seed)
    ids = list(range(n))
    pairs = list(itertools.combinations(ids, 2))
    return rng.sample(pairs, round(len(pairs) * density))


def build(cls, n, edges):
    g = cls(n)
    for u, v in edges:
        g.add_edge(u, v)
    return g


# --------------------------------------------------------------------------
# Обходы
# --------------------------------------------------------------------------
def bfs(graph, start):
    """Обход в ширину. Возвращает (порядок обхода, расстояния, предки)."""
    n = graph.n
    dist = [-1] * n
    parent = [-1] * n
    order = []
    dist[start] = 0
    queue = deque([start])
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in graph.neighbors(v):
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                parent[u] = v
                queue.append(u)
    return order, dist, parent


def bfs_stats(graph, start):
    """Обход в ширину со статистикой: (глубина = число уровней − 1, максимальная длина очереди)."""
    dist = [-1] * graph.n
    dist[start] = 0
    queue = deque([start])
    depth, max_queue = 0, 1
    while queue:
        v = queue.popleft()
        if dist[v] > depth:
            depth = dist[v]
        for u in graph.neighbors(v):
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                queue.append(u)
                if len(queue) > max_queue:
                    max_queue = len(queue)
    return depth, max_queue


def dfs(graph, start, visited=None):
    """Итеративный обход в глубину с явным стеком. Возвращает порядок обхода."""
    if visited is None:
        visited = [False] * graph.n
    order, stack = [], [start]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for u in reversed(graph.neighbors(v)):
            if not visited[u]:
                stack.append(u)
    return order


def dfs_tree(graph, start):
    """
    Итеративный DFS с построением дерева обхода.
    Возвращает (порядок, предки, глубины, максимальный размер стека).
    Глубина дерева обхода = максимальная глубина рекурсии рекурсивной версии.
    """
    n = graph.n
    visited = [False] * n
    parent = [-1] * n
    depth = [0] * n
    order = []
    stack = [(start, -1)]
    max_stack = 1
    while stack:
        v, p = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        parent[v] = p
        depth[v] = depth[p] + 1 if p >= 0 else 0
        order.append(v)
        for u in reversed(graph.neighbors(v)):
            if not visited[u]:
                stack.append((u, v))
        if len(stack) > max_stack:
            max_stack = len(stack)
    return order, parent, depth, max_stack


def dfs_recursive(graph, start):
    """Рекурсивный DFS — для сравнения; глубина рекурсии равна глубине дерева обхода."""
    visited = [False] * graph.n
    order = []

    def visit(v):
        visited[v] = True
        order.append(v)
        for u in graph.neighbors(v):
            if not visited[u]:
                visit(u)

    visit(start)
    return order


def connected_components(graph):
    """Компоненты связности повторными запусками BFS (полный обход графа)."""
    seen = [False] * graph.n
    components = []
    for s in range(graph.n):
        if seen[s]:
            continue
        seen[s] = True
        queue = deque([s])
        comp = []
        while queue:
            v = queue.popleft()
            comp.append(v)
            for u in graph.neighbors(v):
                if not seen[u]:
                    seen[u] = True
                    queue.append(u)
        components.append(comp)
    return components


def components_dfs(graph):
    """Компоненты связности повторными запусками итеративного DFS (полный обход графа)."""
    visited = [False] * graph.n
    return [dfs(graph, s, visited) for s in range(graph.n) if not visited[s]]


# --------------------------------------------------------------------------
# Вспомогательные функции измерений
# --------------------------------------------------------------------------
def measure(action, repeats=REPEATS):
    """Медианное время action(), с. Сборщик мусора на время замера отключён."""
    times = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        action()
        times.append(time.perf_counter() - t0)
        gc.enable()
    return statistics.median(times)


def deep_sizeof(obj):
    """Расход памяти вложенной структуры списков, байт.
    Каждый объект учитывается один раз (по id): общие числа не считаются повторно."""
    seen, total, stack = set(), 0, [obj]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        total += sys.getsizeof(o)
        if isinstance(o, list):
            unique = dict(zip(map(id, o), o))        # уникальные элементы без цикла на Python
            stack.extend(x for i, x in unique.items() if i not in seen)
    return total


def peak_memory(action):
    """Пиковый дополнительный расход памяти при выполнении action (tracemalloc), байт."""
    gc.collect()
    tracemalloc.start()
    action()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def est_matrix_bytes(n, bytes_cells):
    """Расчётный размер матрицы n × n: строки + список строк (по sys.getsizeof)."""
    if bytes_cells:
        return n * sys.getsizeof(bytearray(n)) + sys.getsizeof([None] * n)
    return n * sys.getsizeof([0] * n) + sys.getsizeof([None] * n)


def mb(x):
    return x / 1024 / 1024


# --------------------------------------------------------------------------
# Проверка корректности (тестовый граф и случайные тесты)
# --------------------------------------------------------------------------
TEST_N = 10
TEST_EDGES = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (8, 10)]


def experiment_small():
    """Тестовый граф из 10 вершин (две компоненты): порядок обхода и деревья обхода."""
    result = {"n": TEST_N, "edges": TEST_EDGES, "graphs": {}}
    for cls in (MatrixGraph, ListGraph):
        g = cls(TEST_N)
        for u, v in TEST_EDGES:
            g.add_edge(u - 1, v - 1)
        order, dist, parent = bfs(g, 0)
        d_order, d_parent, d_depth, d_stack = dfs_tree(g, 0)
        result["graphs"][cls.__name__] = {
            "bfs_order": [v + 1 for v in order],
            "bfs_dist": {str(v + 1): d for v, d in enumerate(dist)},
            "bfs_tree": [[parent[v] + 1, v + 1] for v in order if parent[v] >= 0],
            "dfs_order": [v + 1 for v in d_order],
            "dfs_tree": [[d_parent[v] + 1, v + 1] for v in d_order if d_parent[v] >= 0],
            "dfs_depth": max(d_depth[v] for v in d_order),
            "components": [sorted(v + 1 for v in c) for c in connected_components(g)],
        }
    return result


def check_correctness(trials=300):
    """Случайные тесты: оба представления дают одинаковые результаты, которые совпадают
    с независимыми расчётами (Флойд — Уоршелл для расстояний, система непересекающихся
    множеств для компонент); рекурсивный и итеративный DFS дают один порядок обхода."""
    rng = random.Random(SEED)
    for _ in range(trials):
        n = rng.randint(1, 12)
        pairs = list(itertools.combinations(range(n), 2))
        edges = rng.sample(pairs, rng.randint(0, len(pairs)))
        gm, gl = build(MatrixGraph, n, edges), build(ListGraph, n, edges)

        inf = float("inf")
        fw = [[0 if i == j else inf for j in range(n)] for i in range(n)]
        for u, v in edges:
            fw[u][v] = fw[v][u] = 1
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if fw[i][k] + fw[k][j] < fw[i][j]:
                        fw[i][j] = fw[i][k] + fw[k][j]

        root = list(range(n))

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        for u, v in edges:
            root[find(u)] = find(v)
        groups = {}
        for v in range(n):
            groups.setdefault(find(v), []).append(v)
        expected_components = sorted(sorted(c) for c in groups.values())

        for g in (gm, gl):
            for s in range(n):
                _, dist, _ = bfs(g, s)
                assert dist == [fw[s][t] if fw[s][t] != inf else -1 for t in range(n)]
                assert dfs(g, s) == dfs_recursive(g, s) == dfs_tree(g, s)[0]
            assert sorted(sorted(c) for c in connected_components(g)) == expected_components
            assert sorted(sorted(c) for c in components_dfs(g)) == expected_components
        assert all(gm.neighbors(v) == sorted(gl.neighbors(v)) for v in range(n))
        assert all(gm.has_edge(u, v) == gl.has_edge(u, v) for u, v in pairs)
    # цепочка: глубина и размеры структур известны аналитически
    for n in (1, 2, 10, 1000):
        g = make_chain(ListGraph, n)
        assert bfs_stats(g, 0) == (n - 1, 1)
        order, _, depth, stack = dfs_tree(g, 0)
        assert order == list(range(n)) and max(depth) == n - 1 and stack <= 2
    return True


# --------------------------------------------------------------------------
# Задача варианта: цепочка — глубина обхода и память
# --------------------------------------------------------------------------
def experiment_chain():
    res = {"lists": [], "middle": [], "matrix": [], "estimates": []}

    # --- списки смежности: все размеры ряда ---
    for n in EXT_SIZES:
        t0 = time.perf_counter()
        g = make_chain(ListGraph, n)
        t_build = time.perf_counter() - t0
        depth_b, queue = bfs_stats(g, 0)
        order, _, depth_d, stack = dfs_tree(g, 0)
        assert depth_b == n - 1 and len(order) == n
        mem = deep_sizeof(g.adj)
        res["lists"].append({
            "n": n, "m": g.m,
            "t_build": t_build,
            "t_bfs": measure(lambda: bfs(g, 0)),
            "t_dfs": measure(lambda: dfs(g, 0)),
            "mem": mem, "per_vertex": mem / n,
            "bfs_depth": depth_b, "bfs_queue": queue,
            "dfs_depth": max(depth_d), "dfs_stack": stack,
            "peak_bfs": peak_memory(lambda: bfs(g, 0)),
            "peak_dfs": peak_memory(lambda: dfs(g, 0)),
        })

    # --- та же цепочка, но старт из середины ---
    for n in SIZES:
        g = make_chain(ListGraph, n)
        s = n // 2
        depth_b, queue = bfs_stats(g, s)
        _, _, depth_d, stack = dfs_tree(g, s)
        res["middle"].append({"n": n, "start": s, "bfs_depth": depth_b, "bfs_queue": queue,
                              "dfs_depth": max(depth_d), "dfs_stack": stack})

    # --- матрица смежности: реально строится только для небольших n ---
    for n in MATRIX_SIZES:
        reps = 3 if n >= 2000 else REPEATS
        gm = make_chain(MatrixGraph, n)
        gb = make_chain(MatrixGraphBytes, n)
        mem_list = deep_sizeof(gm.matrix)
        mem_bytes = deep_sizeof(gb.matrix)
        res["matrix"].append({
            "n": n,
            "t_bfs": measure(lambda: bfs(gm, 0), reps),
            "t_dfs": measure(lambda: dfs(gm, 0), reps),
            "mem_list": mem_list, "mem_bytes": mem_bytes,
            "est_list": est_matrix_bytes(n, False), "est_bytes": est_matrix_bytes(n, True),
        })
        del gm, gb
        gc.collect()

    # --- расчётные оценки для размеров варианта ---
    ref = res["matrix"][-1]                                   # n = 5000: опорная точка по времени
    by_n = {r["n"]: r for r in res["lists"]}
    for n in SIZES:
        scale = (n / ref["n"]) ** 2
        res["estimates"].append({
            "n": n,
            "mem_list": est_matrix_bytes(n, False),
            "mem_bytes": est_matrix_bytes(n, True),
            "mem_lists": by_n[n]["mem"],
            "t_bfs_matrix": ref["t_bfs"] * scale, "t_dfs_matrix": ref["t_dfs"] * scale,
            "t_bfs_lists": by_n[n]["t_bfs"], "t_dfs_lists": by_n[n]["t_dfs"],
            "measured": n in MATRIX_SIZES,
        })
    # порядок роста времени матрицы: T(2n)/T(n) по измеренным точкам
    mt = res["matrix"]
    res["matrix_ratios"] = {
        "bfs": [mt[i + 1]["t_bfs"] / mt[i]["t_bfs"] for i in range(len(mt) - 1)],
        "dfs": [mt[i + 1]["t_dfs"] / mt[i]["t_dfs"] for i in range(len(mt) - 1)],
        "sizes": [(mt[i]["n"], mt[i + 1]["n"]) for i in range(len(mt) - 1)],
    }
    lt = res["lists"]
    res["lists_ratios"] = {
        "bfs": [lt[i + 1]["t_bfs"] / lt[i]["t_bfs"] for i in range(len(lt) - 1)],
        "sizes": [(lt[i]["n"], lt[i + 1]["n"]) for i in range(len(lt) - 1)],
    }
    return res


# --------------------------------------------------------------------------
# Рекурсивный и итеративный DFS на цепочке: предел рекурсии
# --------------------------------------------------------------------------
def run_probe(mode, n, limit):
    """Дочерний процесс: строит цепочку и обходит её рекурсивным или итеративным DFS.
    Результат выводится одной строкой JSON. Отдельный процесс нужен, чтобы возможный
    аварийный отказ интерпретатора не прервал основную программу."""
    sys.setrecursionlimit(limit)
    threading.stack_size(512 * 1024 * 1024)          # большой стек потока (для старых версий Python)
    result = {}

    def work():
        try:
            g = make_chain(ListGraph, n)
            t0 = time.perf_counter()
            order = dfs_recursive(g, 0) if mode == "recursive" else dfs(g, 0)
            result.update(ok=len(order) == n, time=time.perf_counter() - t0)
        except RecursionError:
            result.update(ok=False, error="RecursionError")
        except MemoryError:
            result.update(ok=False, error="MemoryError")

    th = threading.Thread(target=work)
    th.start()
    th.join()
    print(json.dumps(result))


def probe(mode, n, limit):
    """Запуск пробы в дочернем процессе."""
    try:
        p = subprocess.run([sys.executable, __file__, "--probe", mode, str(n), str(limit)],
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "превышено время ожидания"}
    if p.returncode != 0 or not p.stdout.strip():
        return {"ok": False, "error": f"аварийное завершение процесса (код {p.returncode})"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def recursion_threshold():
    """Наибольшее n, при котором рекурсивный DFS на цепочке работает при стандартном пределе рекурсии."""
    lo, hi = 10, 20_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            dfs_recursive(make_chain(ListGraph, mid), 0)
            lo = mid
        except RecursionError:
            hi = mid - 1
    return lo


def experiment_recursion():
    default_limit = sys.getrecursionlimit()
    res = {"default_limit": default_limit, "threshold": recursion_threshold(), "runs": []}
    for n in SIZES:
        entry = {"n": n,
                 "recursive_default": probe("recursive", n, default_limit),
                 "recursive_raised": probe("recursive", n, n + 1000),
                 "iterative": probe("iterative", n, default_limit)}
        res["runs"].append(entry)
    return res


# --------------------------------------------------------------------------
# Дополнительное задание И6: зависимость времени и памяти от плотности
# --------------------------------------------------------------------------
def experiment_density():
    n = DENSITY_N
    max_edges = n * (n - 1) // 2
    rows = []
    for d in DENSITIES:
        edges = random_edges(n, d)
        gm = build(MatrixGraph, n, edges)
        gb = build(MatrixGraphBytes, n, edges)
        gl = build(ListGraph, n, edges)
        rng = random.Random(SEED)
        queries = [(rng.randrange(n), rng.randrange(n)) for _ in range(HAS_EDGE_QUERIES)]

        def check_edges(g):
            for u, v in queries:
                g.has_edge(u, v)

        comps = len(connected_components(gl))
        assert len(connected_components(gm)) == comps == len(components_dfs(gl))
        rows.append({
            "density": d, "m": len(edges), "components": comps,
            "t_bfs_matrix": measure(lambda: connected_components(gm)),
            "t_bfs_lists": measure(lambda: connected_components(gl)),
            "t_dfs_matrix": measure(lambda: components_dfs(gm)),
            "t_dfs_lists": measure(lambda: components_dfs(gl)),
            "t_has_matrix": measure(lambda: check_edges(gm)),
            "t_has_lists": measure(lambda: check_edges(gl)),
            "mem_matrix_list": deep_sizeof(gm.matrix),
            "mem_matrix_bytes": deep_sizeof(gb.matrix),
            "mem_lists": deep_sizeof(gl.adj),
        })
        del gm, gb, gl
        gc.collect()

    # Память списков линейна по m: mem = base + c·m. Находим плотность, при которой
    # матрица занимает не больше места, чем списки.
    first, last = rows[0], rows[-1]
    c = (last["mem_lists"] - first["mem_lists"]) / (last["m"] - first["m"])
    base = first["mem_lists"] - c * first["m"]
    crossover = {}
    for key in ("mem_matrix_list", "mem_matrix_bytes"):
        m_star = (rows[0][key] - base) / c
        crossover[key] = m_star / max_edges            # доля от максимума; > 1 — пересечения нет
    # по времени: первая плотность, где матрица не медленнее списков
    time_cross = {}
    for algo in ("bfs", "dfs"):
        hit = [r["density"] for r in rows if r[f"t_{algo}_matrix"] <= r[f"t_{algo}_lists"]]
        time_cross[algo] = hit[0] if hit else None
    hit = [r["density"] for r in rows if r["t_has_matrix"] <= r["t_has_lists"]]
    time_cross["has_edge"] = hit[0] if hit else None
    return {"n": n, "max_edges": max_edges, "rows": rows, "per_edge_bytes": c,
            "base_bytes": base, "crossover_memory": crossover, "crossover_time": time_cross}


# --------------------------------------------------------------------------
# Печать результатов
# --------------------------------------------------------------------------
def fmt(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def print_report(r):
    s = r["small"]["graphs"]["ListGraph"]
    print("=" * 96)
    print("1. Тестовый граф (10 вершин, 10 рёбер): " + ", ".join(f"{u}–{v}" for u, v in TEST_EDGES))
    print("=" * 96)
    print("BFS от вершины 1:   ", s["bfs_order"])
    print("Расстояния:         ", s["bfs_dist"])
    print("Дерево BFS (предок→вершина):", s["bfs_tree"])
    print("DFS от вершины 1:   ", s["dfs_order"])
    print("Дерево DFS (предок→вершина):", s["dfs_tree"], f"(глубина {s['dfs_depth']})")
    print("Компоненты связности:", s["components"])
    same = r["small"]["graphs"]["MatrixGraph"] == r["small"]["graphs"]["ListGraph"]
    print("Матрица и списки дали одинаковый результат:", same)

    print("\n" + "=" * 96)
    print("2. Цепочка: списки смежности (старт в вершине 0)")
    print("=" * 96)
    print(f"{'n':>8}{'построение':>12}{'BFS, с':>10}{'DFS, с':>10}{'память, МБ':>12}{'байт/верш.':>12}"
          f"{'глуб.BFS':>10}{'очередь':>9}{'глуб.DFS':>10}{'стек':>6}{'пик BFS,МБ':>12}{'пик DFS,МБ':>12}")
    for x in r["chain"]["lists"]:
        print(f"{x['n']:>8}{fmt(x['t_build']):>12}{fmt(x['t_bfs']):>10}{fmt(x['t_dfs']):>10}"
              f"{fmt(mb(x['mem']), 2):>12}{fmt(x['per_vertex'], 1):>12}{x['bfs_depth']:>10}{x['bfs_queue']:>9}"
              f"{x['dfs_depth']:>10}{x['dfs_stack']:>6}{fmt(mb(x['peak_bfs']), 2):>12}{fmt(mb(x['peak_dfs']), 2):>12}")
    print("\nСтарт в середине цепочки:")
    print(f"{'n':>8}{'старт':>8}{'глуб.BFS':>10}{'очередь':>9}{'глуб.DFS':>10}{'стек':>6}")
    for x in r["chain"]["middle"]:
        print(f"{x['n']:>8}{x['start']:>8}{x['bfs_depth']:>10}{x['bfs_queue']:>9}{x['dfs_depth']:>10}{x['dfs_stack']:>6}")

    print("\n" + "=" * 96)
    print("3. Цепочка: матрица смежности (построена реально)")
    print("=" * 96)
    print(f"{'n':>8}{'BFS, с':>10}{'DFS, с':>10}{'список,МБ':>12}{'bytearray,МБ':>14}{'расчёт спис.':>14}{'расчёт bytes':>14}")
    for x in r["chain"]["matrix"]:
        print(f"{x['n']:>8}{fmt(x['t_bfs']):>10}{fmt(x['t_dfs']):>10}{fmt(mb(x['mem_list']), 2):>12}"
              f"{fmt(mb(x['mem_bytes']), 2):>14}{fmt(mb(x['est_list']), 2):>14}{fmt(mb(x['est_bytes']), 2):>14}")
    mr = r["chain"]["matrix_ratios"]
    print("Рост времени BFS по матрице:", ", ".join(
        f"{a}→{b}: ×{fmt(v, 2)}" for (a, b), v in zip(mr["sizes"], mr["bfs"])))
    lr = r["chain"]["lists_ratios"]
    print("Рост времени BFS по спискам:", ", ".join(
        f"{a}→{b}: ×{fmt(v, 2)}" for (a, b), v in zip(lr["sizes"], lr["bfs"])))

    print("\n" + "=" * 96)
    print("4. Размеры варианта: матрица (расчёт; построена только для n = 5000) и списки")
    print("=" * 96)
    print(f"{'n':>8}{'матр.(спис.),МБ':>17}{'матр.(bytes),МБ':>17}{'списки, МБ':>12}{'отношение':>11}"
          f"{'BFS матр.,с*':>14}{'BFS спис.,с':>13}")
    for x in r["chain"]["estimates"]:
        print(f"{x['n']:>8}{fmt(mb(x['mem_list']), 1):>17}{fmt(mb(x['mem_bytes']), 1):>17}"
              f"{fmt(mb(x['mem_lists']), 2):>12}{fmt(x['mem_list'] / x['mem_lists'], 0):>11}"
              f"{fmt(x['t_bfs_matrix'], 2):>14}{fmt(x['t_bfs_lists']):>13}")
    print("* для n > 5000 время по матрице — экстраполяция по закону Θ(n²) от измеренного при n = 5000")

    rc = r["recursion"]
    print("\n" + "=" * 96)
    print(f"5. Рекурсивный и итеративный DFS на цепочке (стандартный предел рекурсии: {rc['default_limit']})")
    print("=" * 96)
    print(f"Наибольшая цепочка, которую обходит рекурсивный DFS при стандартном пределе: n = {rc['threshold']}")

    def desc(p):
        if p.get("ok"):
            return f"работает, {fmt(p['time'], 3)} с"
        return f"сбой: {p.get('error')}"

    for x in rc["runs"]:
        print(f"n = {x['n']}:")
        print(f"   рекурсивный, стандартный предел  — {desc(x['recursive_default'])}")
        print(f"   рекурсивный, предел n + 1000     — {desc(x['recursive_raised'])}")
        print(f"   итеративный (явный стек)         — {desc(x['iterative'])}")

    d = r["density"]
    print("\n" + "=" * 96)
    print(f"6. И6: время полного обхода и память при n = {d['n']} в зависимости от плотности")
    print("=" * 96)
    print(f"{'доля':>6}{'m':>9}{'BFS матр.':>11}{'BFS спис.':>11}{'DFS матр.':>11}{'DFS спис.':>11}"
          f"{'матр.(спис),МБ':>16}{'матр.(bytes),МБ':>17}{'списки, МБ':>12}")
    for x in d["rows"]:
        print(f"{x['density'] * 100:>5.0f}%{x['m']:>9}{fmt(x['t_bfs_matrix']):>11}{fmt(x['t_bfs_lists']):>11}"
              f"{fmt(x['t_dfs_matrix']):>11}{fmt(x['t_dfs_lists']):>11}{fmt(mb(x['mem_matrix_list']), 2):>16}"
              f"{fmt(mb(x['mem_matrix_bytes']), 2):>17}{fmt(mb(x['mem_lists']), 2):>12}")
    print(f"Память списков ≈ {fmt(mb(d['base_bytes']), 2)} МБ + {fmt(d['per_edge_bytes'], 1)} байт на ребро")
    for key, label in (("mem_matrix_list", "матрица (список списков)"), ("mem_matrix_bytes", "матрица (bytearray)")):
        v = d["crossover_memory"][key]
        txt = "пересечения нет (матрица не выгоднее даже при 100 %)" if v > 1 else f"{v * 100:.1f} % рёбер от максимума"
        print(f"Точка, где {label} занимает меньше места, чем списки: {txt}")
    print(f"\nПроверка наличия ребра ({HAS_EDGE_QUERIES} запросов), с:")
    print(f"{'доля':>6}{'матрица':>11}{'списки':>11}")
    for x in d["rows"]:
        print(f"{x['density'] * 100:>5.0f}%{fmt(x['t_has_matrix']):>11}{fmt(x['t_has_lists']):>11}")
    for key, label in (("bfs", "полный обход BFS"), ("dfs", "полный обход DFS"), ("has_edge", "проверка has_edge")):
        v = d["crossover_time"][key]
        print(f"Время ({label}): матрица не медленнее списков " + (f"начиная с {v * 100:.0f} %" if v else "ни при одной из плотностей"))


# --------------------------------------------------------------------------
# Графики (matplotlib подключается только при запросе)
# --------------------------------------------------------------------------
def save_plots(r, outdir):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    ch, dn = r["chain"], r["density"]
    BL, RD, GR = "#1f77b4", "#c0392b", "#2a9d8f"

    # --- 1. время обхода цепочки от n ---
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ns = [x["n"] for x in ch["lists"]]
    ax.plot(ns, [x["t_bfs"] for x in ch["lists"]], "o-", color=BL, label="Списки смежности: BFS")
    ax.plot(ns, [x["t_dfs"] for x in ch["lists"]], "s-", color=GR, label="Списки смежности: DFS")
    mns = [x["n"] for x in ch["matrix"]]
    ax.plot(mns, [x["t_bfs"] for x in ch["matrix"]], "o-", color=RD, label="Матрица смежности: BFS (измерено)")
    ax.plot(mns, [x["t_dfs"] for x in ch["matrix"]], "s-", color="#e08e79", label="Матрица смежности: DFS (измерено)")
    ref = ch["matrix"][-1]
    ext = [ref["n"]] + [n for n in SIZES if n > ref["n"]]
    ax.plot(ext, [ref["t_bfs"] * (n / ref["n"]) ** 2 for n in ext], "--", color=RD, label="Матрица: BFS, экстраполяция Θ(n²)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число вершин n")
    ax.set_ylabel("Время полного обхода, с")
    ax.set_title("Цепочка: время обхода")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chain_time.png"), dpi=200)
    plt.close(fig)

    # --- 2. память: списки и матрица ---
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    w = 0.26
    est = ch["estimates"]
    xs = range(len(est))
    series = (("Матрица (список списков)", "mem_list", RD), ("Матрица (bytearray)", "mem_bytes", "#e08e79"),
              ("Списки смежности", "mem_lists", BL))
    for k, (label, key, col) in enumerate(series):
        vals = [mb(e[key]) for e in est]
        bars = ax.bar([x + (k - 1) * w for x in xs], vals, w, color=col, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:,.0f}".replace(",", " ") if v >= 100 else f"{v:.1f}",
                    ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"n = {e['n']}" for e in est])
    ax.set_ylabel("Память, МБ (логарифмическая шкала)")
    ax.set_title("Цепочка: память представления (для матрицы при n > 5000 — расчёт)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chain_memory.png"), dpi=200)
    plt.close(fig)

    # --- 3. И6: время от плотности ---
    pct = [x["density"] * 100 for x in dn["rows"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(pct, [x["t_bfs_matrix"] for x in dn["rows"]], "o-", color=RD, label="Матрица: BFS")
    ax.plot(pct, [x["t_bfs_lists"] for x in dn["rows"]], "o-", color=BL, label="Списки: BFS")
    ax.plot(pct, [x["t_dfs_matrix"] for x in dn["rows"]], "s--", color="#e08e79", label="Матрица: DFS")
    ax.plot(pct, [x["t_dfs_lists"] for x in dn["rows"]], "s--", color=GR, label="Списки: DFS")
    ax.set_xscale("log")
    ax.set_xticks(pct)
    ax.set_xticklabels([f"{p:g} %" for p in pct])
    ax.set_xlabel("Плотность: доля рёбер от максимума")
    ax.set_ylabel("Время полного обхода, с")
    ax.set_title(f"Время обхода в зависимости от плотности, n = {dn['n']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "density_time.png"), dpi=200)
    plt.close(fig)

    # --- 4. И6: память от плотности ---
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(pct, [mb(x["mem_matrix_list"]) for x in dn["rows"]], "o-", color=RD, label="Матрица (список списков)")
    ax.plot(pct, [mb(x["mem_matrix_bytes"]) for x in dn["rows"]], "o-", color="#e08e79", label="Матрица (bytearray)")
    ax.plot(pct, [mb(x["mem_lists"]) for x in dn["rows"]], "o-", color=BL, label="Списки смежности")
    cm = dn["crossover_memory"]["mem_matrix_bytes"] * 100
    if 0 < cm <= 100:
        ax.axvline(cm, color="gray", ls=":")
        ax.text(cm * 1.05, ax.get_ylim()[0] * 1.5, f"{cm:.1f} %", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(pct)
    ax.set_xticklabels([f"{p:g} %" for p in pct])
    ax.set_xlabel("Плотность: доля рёбер от максимума")
    ax.set_ylabel("Память, МБ")
    ax.set_title(f"Память в зависимости от плотности, n = {dn['n']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "density_memory.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ЛР № 9, вариант 13: представления графов и обходы")
    parser.add_argument("--plots", metavar="DIR", help="сохранить графики в каталог DIR")
    parser.add_argument("--json", metavar="FILE", help="сохранить результаты в JSON")
    parser.add_argument("--probe", nargs=3, metavar=("MODE", "N", "LIMIT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.probe:                                   # служебный режим: дочерний процесс
        run_probe(args.probe[0], int(args.probe[1]), int(args.probe[2]))
        return

    check_correctness()
    print("Случайные тесты (300 графов до 12 вершин; цепочки): пройдены\n")
    results = {"small": experiment_small()}
    print("Выполняются замеры (около минуты)…\n")
    results["chain"] = experiment_chain()
    results["recursion"] = experiment_recursion()
    results["density"] = experiment_density()
    print_report(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    if args.plots:
        save_plots(results, args.plots)
        print(f"\nГрафики сохранены в каталог «{args.plots}»")


if __name__ == "__main__":
    main()
