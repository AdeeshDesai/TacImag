"""Hydra SearchPathPlugin: expose ManiFeel's config directory.

ManiFeel's env runner composes its IsaacGym config (isaacgym_config_*.yaml
plus the task/TacSLTask*.yaml group) from the hydra search path at rollout
time. ManiFeel's config dir is plain files inside the installed package (not
a Python package itself), so this plugin locates it via the manifeel module
and appends it — Hydra auto-discovers any plugin under the `hydra_plugins`
namespace, and a plugin-provided search path persists for every compose call
in the process (unlike a per-run `hydra.searchpath` override).

No-op when manifeel is not installed (simulator-free stage-1 setups).
"""
import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class TacImagSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        try:
            import manifeel
        except ImportError:
            return
        # __path__ covers both regular and namespace-package installs
        for pkg_dir in list(getattr(manifeel, '__path__', [])):
            cfg_dir = os.path.join(pkg_dir, 'config')
            if os.path.isdir(cfg_dir):
                search_path.append(
                    provider='tacimag', path=f'file://{cfg_dir}')
                return
