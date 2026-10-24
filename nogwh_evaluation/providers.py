"""
Three VLM providers wired with native multimodal few-shot (P6: 5 example images + target).
- OpenAI:      multimodal chat message with image_url parts
- Gemini:      content list with multiple inline Parts
- Together AI: OpenAI-compatible chat with multiple image_url parts
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("providers")

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY   = os.getenv("GOOGLE_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_AI_API_KEY") or os.getenv("TOGETHER_API_KEY")


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mime_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


# ----------------------------- OpenAI -----------------------------

class OpenAIChat:
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=OPENAI_KEY)

    async def few_shot(self, prompt: str, examples: list[tuple[str, int, str]], target: str) -> str | None:
        """
        examples: list of (image_path, score, description) tuples
        target:   path to target image to score
        """
        content: list[dict] = [{"type": "text", "text": prompt}]
        for path, score, desc in examples:
            content.append({"type": "text", "text": f"\n--- EXAMPLE (Score {score}) ---\n{desc}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_for(path)};base64,{b64(path)}"},
            })
        content.append({"type": "text", "text": "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_for(target)};base64,{b64(target)}"},
        })

        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        is_gpt5 = "gpt-5" in self.model.lower()
        if is_gpt5:
            params["max_completion_tokens"] = 8000
            params["temperature"] = 1.0
            if "5.2" in self.model or "5.1" in self.model:
                params["reasoning_effort"] = "low"
        else:
            params["max_tokens"] = 8000
            params["temperature"] = 0.0

        try:
            resp = await self.client.chat.completions.create(**params)
            return resp.choices[0].message.content
        except Exception as e:
            log.error("OpenAI error (%s): %s", self.model, e)
            return None


# ----------------------------- Google Gemini -----------------------------

class GeminiChat:
    def __init__(self, model: str):
        self.model = model
        try:
            from google import genai
            from google.genai import types  # noqa: F401
            self._genai = genai
            self._client = genai.Client(api_key=GOOGLE_KEY)
            self._new = True
        except Exception:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_KEY)
            self._genai = genai
            self._model_obj = genai.GenerativeModel(model)
            self._new = False

    async def few_shot(self, prompt: str, examples: list[tuple[str, int, str]], target: str) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_call, prompt, examples, target)

    def _sync_call(self, prompt, examples, target):
        import time
        try:
            if self._new:
                from google.genai import types

                def build_parts():
                    parts: list = [types.Part.from_text(text=prompt)]
                    for path, score, desc in examples:
                        parts.append(types.Part.from_text(text=f"\n--- EXAMPLE (Score {score}) ---\n{desc}"))
                        with open(path, "rb") as f:
                            parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_for(path)))
                    parts.append(types.Part.from_text(text="\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."))
                    with open(target, "rb") as f:
                        parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_for(target)))
                    return parts

                # Disable safety filters (residential photos sometimes trip them)
                safety_settings = [
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]

                for attempt in range(3):
                    try:
                        resp = self._client.models.generate_content(
                            model=self.model,
                            contents=[types.Content(role="user", parts=build_parts())],
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                max_output_tokens=8000,
                                safety_settings=safety_settings,
                            ),
                        )
                        text = (resp.text or "").strip() if hasattr(resp, "text") else ""
                        if text:
                            return text
                        log.warning("Gemini empty response (attempt %d). finish_reason=%s", attempt + 1, getattr(resp, "candidates", [{}])[0] if hasattr(resp, "candidates") else "?")
                        time.sleep(1.5 * (attempt + 1))
                    except Exception as inner:
                        log.warning("Gemini retry %d failed: %s", attempt + 1, inner)
                        time.sleep(1.5 * (attempt + 1))
                return None
            else:
                import PIL.Image
                segments: list = [prompt]
                for path, score, desc in examples:
                    segments.append(f"\n--- EXAMPLE (Score {score}) ---\n{desc}")
                    segments.append(PIL.Image.open(path))
                segments.append("\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response.")
                segments.append(PIL.Image.open(target))
                resp = self._model_obj.generate_content(
                    segments,
                    generation_config={"temperature": 0.0, "max_output_tokens": 4000},
                )
                return resp.text
        except Exception as e:
            log.error("Gemini error (%s): %s", self.model, e)
            return None


# ----------------------------- Together AI (Qwen3-VL) -----------------------------

def _composite_examples(examples: list[tuple[str, int, str]], cell: int = 448) -> str:
    """Tile the few-shot example images into ONE labeled montage (base64 PNG).
    Used for providers that cap the number of images per request (e.g. Together's
    Qwen3.5 allows max 5 images, but P6 needs 5 examples + 1 target = 6)."""
    from PIL import Image, ImageDraw, ImageFont
    label_h = 28
    tiles = []
    for path, score, _ in examples:
        im = Image.open(path).convert("RGB")
        im.thumbnail((cell, cell))
        canvas = Image.new("RGB", (cell, cell + label_h), (255, 255, 255))
        canvas.paste(im, ((cell - im.width) // 2, label_h + (cell - im.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, cell, label_h], fill=(20, 20, 20))
        d.text((6, 6), f"EXAMPLE  Score {score}", fill=(255, 255, 255))
        tiles.append(canvas)
    cols = len(tiles)
    grid = Image.new("RGB", (cols * cell, cell + label_h), (255, 255, 255))
    for i, t in enumerate(tiles):
        grid.paste(t, (i * cell, 0))
    import io
    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TogetherChat:
    URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, model: str, enable_thinking: bool | None = None, max_images: int = 5):
        self.model = model
        # Together serverless caps images per request (Qwen3.5-9B = 5). P6 needs 6
        # (5 examples + target), so when over the cap we tile the examples into one
        # composite montage to preserve all 5 score anchors.
        self.max_images = max_images
        # Qwen3.5 family runs in "thinking" mode by default, which burns the token
        # budget on reasoning and can return empty content. Default it OFF for those
        # models so we get a fast, direct JSON answer. Pass enable_thinking explicitly
        # to override.
        if enable_thinking is None and "qwen3.5" in model.lower():
            enable_thinking = False
        self.enable_thinking = enable_thinking

    async def few_shot(self, prompt: str, examples: list[tuple[str, int, str]], target: str) -> str | None:
        content: list[dict] = [{"type": "text", "text": prompt}]
        # +1 for the target image
        if len(examples) + 1 > self.max_images:
            # Tile examples into one montage to stay within the per-request image cap.
            content.append({"type": "text", "text":
                "\nThe 5 official example images are provided together in ONE labeled montage "
                "below (left-to-right: Score 1 .. Score 5). Their descriptions:"})
            for _, score, desc in examples:
                content.append({"type": "text", "text": f"\n--- EXAMPLE (Score {score}) ---\n{desc}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_composite_examples(examples)}"},
            })
        else:
            for path, score, desc in examples:
                content.append({"type": "text", "text": f"\n--- EXAMPLE (Score {score}) ---\n{desc}"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_for(path)};base64,{b64(path)}"},
                })
        content.append({"type": "text", "text": "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_for(target)};base64,{b64(target)}"},
        })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8000,
            "temperature": 0.0,
            "top_p": 0.95,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        headers = {"Authorization": f"Bearer {TOGETHER_KEY}", "Content-Type": "application/json"}

        timeout = aiohttp.ClientTimeout(total=120)
        backoffs = [3, 10, 25]
        for attempt in range(len(backoffs) + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.post(self.URL, json=payload, headers=headers) as r:
                        if r.status == 200:
                            data = await r.json()
                            return data["choices"][0]["message"]["content"]
                        text = await r.text()
                        # Retry transient rate-limit / server errors; give up on others.
                        if r.status in (429, 500, 502, 503, 529) and attempt < len(backoffs):
                            log.warning("Together %d (attempt %d), backing off %ds", r.status, attempt + 1, backoffs[attempt])
                            await asyncio.sleep(backoffs[attempt]); continue
                        log.error("Together error %d: %s", r.status, text[:300])
                        return None
            except Exception as e:
                if attempt < len(backoffs):
                    log.warning("Together exception (attempt %d): %s; retrying", attempt + 1, str(e)[:120])
                    await asyncio.sleep(backoffs[attempt]); continue
                log.error("Together error (%s): %s", self.model, e)
                return None
        return None
