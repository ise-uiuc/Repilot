from pathlib import Path
import json
import subprocess
import re


def get_cmd_output(cmd: list[str]) -> str:
    return subprocess.run(cmd, text=True, capture_output=True).stdout


def detect_javahome(output: str) -> str:
    # extract ... from export JAVA_HOME="..."
    # the number of whitespaces is not fixed
    pattern = re.compile(r'export\s+JAVA_HOME="(.*)"')
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise ValueError("Cannot find JAVA_HOME")


java8_home = Path(
    detect_javahome(get_cmd_output(["cs", "java", "--jvm", "8", "--env"]))
)
java18_home = Path(
    detect_javahome(get_cmd_output(["cs", "java", "--jvm", "18", "--env"]))
)

assert java8_home.exists()
assert java18_home.exists()


def get_launcher_path() -> Path:
    root = Path(
        "/root/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/plugins/"
    )
    return next(root.glob("org.eclipse.equinox.launcher_*.jar"))


config_path = Path(
    "/root/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/config_linux"
)

meta_config = {
    "d4j_home": "/root/defects4j",
    "d4j_checkout_root": "/root/d4j-checkout",
    "jdt_ls_repo": "/root/eclipse.jdt.ls",
    "java8_home": str(java8_home.absolute()),
    "language_server_cmd": [
        str((java18_home / "bin" / "java").absolute()),
        "-Declipse.application=org.eclipse.jdt.ls.core.id1",
        "-Dosgi.bundles.defaultStartLevel=4",
        "-Declipse.product=org.eclipse.jdt.ls.core.product",
        "-Dlog.level=ERROR",
        "-noverify",
        "-Xmx1G",
        "--add-modules=ALL-SYSTEM",
        "--add-opens",
        "java.base/java.util=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.lang=ALL-UNNAMED",
        "-jar",
        str(get_launcher_path().absolute()),
        "-configuration",
        str(config_path.absolute()),
    ],
    "seed": 0,
}

Path("/root/Repilot/meta_config.json").write_text(json.dumps(meta_config, indent=2))
