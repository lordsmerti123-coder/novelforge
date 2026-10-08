"""Клиент генерации изображений ComfyUI.

Общение идёт по HTTP API сервера ComfyUI на ``127.0.0.1:8188``. Граф собирается
в **API-формате**: это не содержимое файла воркфлоу из интерфейса, и UI-формат
сервер не примет.

Две ловушки, каждая из которых не даёт ошибки, а тихо портит результат:

* идентификаторы узлов в ссылках — строки (``["4", 0]``); числа дают ``KeyError``
  внутри ``validate_prompt``;
* вход-референс передаётся плоским ключом ``images.image_1``; вложенный словарь
  молча игнорируется, и правка вырождается в генерацию по тексту.

Сервер не поднимает никто, кроме Comfy Desktop: второй экземпляр конфликтует за
порт 8188.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from novel import config

_CREATE_NO_WINDOW = 0x08000000


def _port_busy(base_url: str) -> bool:
    """Занят ли порт сервера хоть кем-нибудь.

    Нужно, чтобы не поднять второй экземпляр поверх чужого: второй сервер
    подерётся за порт и сломает уже работающий.

    @param base_url: адрес сервера.
    @returns: ``True``, если порт принимает соединения.
    """
    spec = urlparse(base_url)
    host = spec.hostname or "127.0.0.1"
    port = spec.port or 8188
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


class ComfyError(RuntimeError):
    """ComfyUI недоступен или отказал в исполнении графа."""


@dataclass
class GenerationResult:
    """Результат одного прогона графа."""

    prompt_id: str
    elapsed_s: float
    images: list[Path] = field(default_factory=list)
    queue_wait_s: float | None = None
    raw_history: dict[str, Any] = field(default_factory=dict)

    @property
    def first_image(self) -> Path | None:
        """Первый готовый файл, если он есть."""
        return self.images[0] if self.images else None


def stage_reference(path: Path) -> str:
    """Кладёт изображение в каталог входов ComfyUI и возвращает имя файла.

    Узел ``LoadImage`` читает только из каталога входов, поэтому образец места
    приходится туда копировать. Имя получает короткий хеш исходного пути: два
    кадра из разных миров могут называться одинаково.

    @param path: путь к изображению-образцу.
    @returns: имя файла, которое понимает ``LoadImage``.
    """
    import hashlib
    import shutil

    config.comfy_input_dir().mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    name = f"{digest}-{path.name}"
    target = config.comfy_input_dir() / name
    if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
        shutil.copy2(path, target)
    return name


def build_t2i_graph(
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 25,
    cfg: float = 1.0,
    filename_prefix: str = "novel",
    clip_name: str | None = None,
    unet_name: str | None = None,
    vae_name: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    """Собирает минимальный граф text-to-image для Qwen-Image-2.1.

    Обязательные для этой модели параметры сэмплинга — ``euler`` и ``simple``
    при ``cfg = 1.0``; при единичном cfg негативный промпт игнорируется.
    Размеры должны быть кратны 32.

    @param prompt: текстовое описание сцены.
    @param seed: зерно; при 0 подставляется текущее время.
    @param reference: имя файла-образца в каталоге входов ComfyUI. Энкодер
        вшивает его в последовательность как VAE-латент, поэтому кадр сохраняет
        облик образца, а композиция задаётся описанием и размером.
    @returns: граф в API-формате.
    """
    models = dict(config.t2i_models())
    if clip_name:
        models["clip_name"] = clip_name
    if unet_name:
        models["unet_name"] = unet_name
    if vae_name:
        models["vae_name"] = vae_name
    graph: dict[str, Any] = {
        "3": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": models["clip_name"], "type": "qwen_image"},
        },
        "4": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": models["unet_name"]},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": models["vae_name"]},
        },
        "6": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["4", 0], "device": "auto", "dtype": "default"},
        },
        "7": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["5", 0],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "resolution": max(width, height),
            },
        },
        "2": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["2", 0],
                "seed": seed or int(time.time()) % 2**31,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix},
        },
    }
    if reference:
        # Образец идёт именно в текст-энкодер: узел вшивает его в последовательность
        # как VAE-латент, поэтому кадр наследует облик образца. Отдельный узел
        # переключения латентов здесь не нужен — размер и композицию задаёт
        # собственный пустой латент.
        graph["1"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        graph["7"]["inputs"]["images.image_1"] = ["1", 0]
    return graph


class ComfyClient:
    """HTTP-клиент сервера ComfyUI."""

    def __init__(self, base_url: str = config.COMFY_BASE_URL, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.client_id = "novelforge"

    # --- транспорт ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ComfyError(f"HTTP {exc.code} на {path}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ComfyError(f"{self.base_url} недоступен: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ComfyError(f"таймаут запроса к {self.base_url}{path}") from exc
        except OSError as exc:
            # Сервер генерации могут остановить посреди работы — например,
            # кнопкой «Стоп всё». Оборванное соединение даёт ConnectionResetError,
            # который не является URLError.
            raise ComfyError(f"соединение с {self.base_url} оборвано: {exc}") from exc
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    # --- наблюдение ---------------------------------------------------------

    def system_stats(self) -> dict[str, Any]:
        """``GET /system_stats`` — версия, устройство, свободная VRAM."""
        return self._request("GET", "/system_stats", timeout_s=10.0)

    def is_alive(self) -> bool:
        """Отвечает ли сервер."""
        try:
            self.system_stats()
            return True
        except (ComfyError, json.JSONDecodeError):
            return False

    def object_info(self, node_class: str) -> dict[str, Any]:
        """``GET /object_info/<node>`` — какие значения принимает узел."""
        return self._request("GET", f"/object_info/{node_class}")

    def available_models(self) -> dict[str, list[str]]:
        """Списки имён, которые видят три загрузчика пайплайна.

        @returns: ключи ``clip_name``, ``unet_name``, ``vae_name``.
        """
        def options(node: str, field: str) -> list[str]:
            doc = self.object_info(node)[node]["input"]["required"][field][0]
            return list(doc) if isinstance(doc, list) else []

        return {
            "clip_name": options("CLIPLoaderGGUF", "clip_name"),
            "unet_name": options("UnetLoaderGGUF", "unet_name"),
            "vae_name": options("VAELoader", "vae_name"),
        }

    def check_models(self) -> dict[str, Any]:
        """Проверяет, что все три модели пайплайна видны загрузчикам.

        @returns: отчёт с флагом ``ok`` и списком отсутствующих файлов.
        """
        available = self.available_models()
        missing = {
            key: value
            for key, value in config.t2i_models().items()
            if value not in available.get(key, [])
        }
        return {"ok": not missing, "missing": missing, "available": available}

    def queue_state(self) -> dict[str, Any]:
        """``GET /queue`` — что стоит в очереди и что исполняется."""
        return self._request("GET", "/queue")

    # --- исполнение ---------------------------------------------------------

    def queue_prompt(self, graph: dict[str, Any]) -> str:
        """Отправляет граф на исполнение.

        @param graph: граф в API-формате.
        @returns: ``prompt_id`` для опроса истории.
        @raises ComfyError: если сервер отверг граф.
        """
        doc = self._request(
            "POST", "/prompt", {"prompt": graph, "client_id": self.client_id}, timeout_s=60.0
        )
        if not doc or "prompt_id" not in doc:
            raise ComfyError(f"сервер не вернул prompt_id: {doc}")
        if doc.get("node_errors"):
            raise ComfyError(f"ошибки узлов: {json.dumps(doc['node_errors'], ensure_ascii=False)}")
        return doc["prompt_id"]

    def history(self, prompt_id: str) -> dict[str, Any]:
        """``GET /history/<prompt_id>``; пустой словарь, пока задачи нет в истории."""
        return self._request("GET", f"/history/{prompt_id}") or {}

    def wait_for(self, prompt_id: str, timeout_s: float = 900.0, poll_s: float = 2.0) -> GenerationResult:
        """Ждёт завершения задачи и собирает пути к готовым файлам.

        @param prompt_id: идентификатор из :meth:`queue_prompt`.
        @param timeout_s: предел ожидания.
        @returns: результат с длительностью и списком файлов.
        @raises ComfyError: если исполнение упало или вышло за отведённое время.
        """
        started = time.time()
        deadline = started + timeout_s
        while time.time() < deadline:
            doc = self.history(prompt_id)
            entry = doc.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                messages = status.get("messages") or []
                for message in messages:
                    if isinstance(message, list) and message and message[0] == "execution_error":
                        raise ComfyError(
                            "ошибка исполнения: "
                            + json.dumps(message[1], ensure_ascii=False)[:600]
                        )
                images: list[Path] = []
                for output in (entry.get("outputs") or {}).values():
                    for image in output.get("images") or []:
                        subfolder = image.get("subfolder") or ""
                        images.append(config.comfy_output_dir() / subfolder / image["filename"])
                return GenerationResult(
                    prompt_id=prompt_id,
                    elapsed_s=time.time() - started,
                    images=images,
                    raw_history=entry,
                )
            time.sleep(poll_s)
        raise ComfyError(f"задача {prompt_id} не завершилась за {timeout_s} c")

    def generate(self, graph: dict[str, Any], timeout_s: float = 900.0) -> GenerationResult:
        """Ставит граф в очередь и ждёт результат."""
        prompt_id = self.queue_prompt(graph)
        return self.wait_for(prompt_id, timeout_s=timeout_s)

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """``POST /free`` — выгружает модели из VRAM.

        Без этого шага ComfyUI держит энкодер и DiT в памяти и не отдаёт VRAM
        обратно движку.
        """
        self._request(
            "POST",
            "/free",
            {"unload_models": unload_models, "free_memory": free_memory},
            timeout_s=60.0,
        )

    # --- запуск сервера -----------------------------------------------------

    def ensure_running(self, timeout_s: float = 300.0, mode: str = "direct") -> tuple[bool, float]:
        """Поднимает сервер ComfyUI, если он не отвечает, и ждёт готовности.

        Два способа запуска:

        * ``direct`` — сервер стартует той же командой, что и у Comfy Desktop,
          только без оболочки. Оркестратор получает полный контроль, и нажатие
          Start в окне приложения не требуется;
        * ``desktop`` — запускается приложение, и дальше нужно нажать Start
          руками; годится, если прямой запуск не подходит.

        Второй сервер не поднимается: перед запуском проверяется, не занят ли
        порт. Если занят кем-то другим, запуск не делается.

        @param timeout_s: сколько ждать ответа ``/system_stats``.
        @param mode: ``direct`` или ``desktop``.
        @returns: ``(запущен ли сервер, сколько секунд ждали)``.
        """
        started = time.time()
        if self.is_alive():
            return True, 0.0
        if _port_busy(self.base_url):
            raise ComfyError(
                f"порт {self.base_url} занят, но сервер не отвечает — "
                "второй экземпляр поднимать нельзя"
            )

        if mode == "desktop":
            if not config.comfy_desktop_exe().exists():
                raise ComfyError(f"не найден {config.comfy_desktop_exe()}")
            subprocess.Popen(
                [str(config.comfy_desktop_exe())],
                cwd=str(config.comfy_desktop_exe().parent),
                creationflags=_CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            )
        else:
            if not config.comfy_server_cwd().is_dir():
                raise ComfyError(f"не найден рабочий каталог {config.comfy_server_cwd()}")
            argv = config.comfy_server_argv()
            # Путь к скрипту задан относительно рабочего каталога сервера, как в
            # исходной команде, поэтому и проверяется от него же.
            missing = [
                item
                for item in argv
                if item.endswith((".py", ".yaml"))
                and not (config.comfy_server_cwd() / item).exists()
                and not Path(item).exists()
            ]
            if missing:
                raise ComfyError("не найдены файлы для прямого запуска: " + ", ".join(missing))
            log_path = config.LOGS_DIR / "comfyui_server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            subprocess.Popen(
                argv,
                cwd=str(config.comfy_server_cwd()),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            )

        deadline = started + timeout_s
        while time.time() < deadline:
            if self.is_alive():
                return True, time.time() - started
            time.sleep(3.0)
        return False, time.time() - started
