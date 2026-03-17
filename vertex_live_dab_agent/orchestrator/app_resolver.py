"""Runtime app target resolver for launch-first execution strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vertex_live_dab_agent.dab.client import DABClientBase


@dataclass
class AppInfo:
    app_id: str
    name: str = ""
    friendly_name: str = ""
    package_name: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedAppTarget:
    app_id: str
    app_name: str = ""
    confidence: float = 0.0
    source: str = ""


class AppResolver:
    """Resolve logical app intent to a launchable app id using runtime catalog."""

    def __init__(self, dab_client: DABClientBase) -> None:
        self._dab = dab_client
        self._catalog_cache: Dict[str, List[AppInfo]] = {}
        self._session_success_map: Dict[str, Dict[str, ResolvedAppTarget]] = {}

    async def load_app_catalog(self, device_id: str, refresh: bool = False) -> List[AppInfo]:
        if device_id in self._catalog_cache and not refresh:
            return self._catalog_cache[device_id]
        resp = await self._dab.list_apps()
        if not resp.success:
            self._catalog_cache[device_id] = []
            return []
        payload = resp.data if isinstance(resp.data, dict) else {}
        apps_raw = payload.get("applications") or payload.get("apps") or []
        catalog: List[AppInfo] = []
        if isinstance(apps_raw, list):
            for item in apps_raw:
                if not isinstance(item, dict):
                    continue
                app_id = str(item.get("appId") or item.get("id") or "").strip()
                if not app_id:
                    continue
                catalog.append(
                    AppInfo(
                        app_id=app_id,
                        name=str(item.get("name") or item.get("label") or "").strip(),
                        friendly_name=str(item.get("friendlyName") or item.get("displayName") or "").strip(),
                        package_name=str(item.get("packageName") or item.get("package") or "").strip(),
                        raw=item,
                    )
                )
        self._catalog_cache[device_id] = catalog
        return catalog

    async def resolve_target_app(
        self,
        goal: str,
        planner_output: Dict[str, Any],
        execution_state: Dict[str, Any],
    ) -> Optional[ResolvedAppTarget]:
        session_id = str(execution_state.get("session_id") or "")
        device_id = str(execution_state.get("device_id") or "default")

        target_name = str(planner_output.get("target_app_name") or "").strip()
        target_domain = str(planner_output.get("target_app_domain") or "").strip().lower()
        target_hint = str(planner_output.get("target_app_hint") or "").strip()
        if not target_name and not target_hint:
            return None

        cache_key = self._cache_key(
            target_app_name=target_name,
            target_app_domain=target_domain,
            target_app_hint=target_hint,
        )
        if session_id:
            by_session = self._session_success_map.get(session_id, {})
            cached = by_session.get(cache_key)
            if cached:
                return cached

        catalog = await self.load_app_catalog(device_id=device_id)
        resolved = self.match_app_candidate(
            target_app_name=target_name,
            target_app_domain=target_domain,
            target_app_hint=target_hint,
            app_catalog=catalog,
        )
        if resolved is None:
            # Refresh from applications/list once more in case catalog changed.
            catalog = await self.load_app_catalog(device_id=device_id, refresh=True)
            resolved = self.match_app_candidate(
                target_app_name=target_name,
                target_app_domain=target_domain,
                target_app_hint=target_hint,
                app_catalog=catalog,
            )
        if resolved and session_id:
            self._session_success_map.setdefault(session_id, {})[cache_key] = resolved
        return resolved

    def match_app_candidate(
        self,
        target_app_name: str,
        target_app_domain: str,
        target_app_hint: str,
        app_catalog: List[AppInfo],
    ) -> Optional[ResolvedAppTarget]:
        name = self._norm(target_app_name)
        hint = self._norm(target_app_hint)
        synonyms = {name, hint}
        if name == "settings" or target_app_domain == "system_settings":
            synonyms.update({"settings", "setting", "device preferences"})
        if name == "youtube" or hint == "youtube":
            synonyms.update({"youtube", "youtube tv"})

        best: Optional[ResolvedAppTarget] = None
        for app in app_catalog:
            app_id_n = self._norm(app.app_id)
            app_name_n = self._norm(app.name)
            app_friendly_n = self._norm(app.friendly_name)
            candidate_text = f"{app_name_n} {app_friendly_n} {app_id_n}"
            score = 0.0
            for syn in [s for s in synonyms if s]:
                if syn == app_id_n:
                    score = max(score, 1.0)
                elif syn == app_friendly_n:
                    score = max(score, 0.99)
                elif syn == app_name_n:
                    score = max(score, 1.0)
                elif syn in candidate_text:
                    score = max(score, 0.85)
                elif self._token_overlap(syn, candidate_text) >= 0.5:
                    score = max(score, 0.75)
            if score > 0 and (best is None or score > best.confidence):
                best = ResolvedAppTarget(
                    app_id=app.app_id,
                    app_name=app.friendly_name or app.name or target_app_name,
                    confidence=score,
                    source="catalog-match",
                )
        return best

    def get_session_resolutions(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for key, value in self._session_success_map.get(session_id, {}).items():
            result[key] = {
                "app_id": value.app_id,
                "app_name": value.app_name,
                "confidence": value.confidence,
                "source": value.source,
            }
        return result

    @staticmethod
    def build_launch_action(
        resolved_target: ResolvedAppTarget,
        launch_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"app_id": resolved_target.app_id}
        if isinstance(launch_parameters, dict):
            for k, v in launch_parameters.items():
                params[str(k)] = v
        return {"action": "LAUNCH_APP", "params": params}

    @staticmethod
    def _cache_key(target_app_name: str, target_app_domain: str, target_app_hint: str) -> str:
        return "|".join(
            [
                AppResolver._norm(target_app_name),
                AppResolver._norm(target_app_domain),
                AppResolver._norm(target_app_hint),
            ]
        )

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(str(value or "").strip().lower().replace("_", " ").split())

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        sa = {t for t in AppResolver._norm(a).split() if t}
        sb = {t for t in AppResolver._norm(b).split() if t}
        if not sa or not sb:
            return 0.0
        return len(sa.intersection(sb)) / float(len(sa))
