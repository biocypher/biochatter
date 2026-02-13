import json
from collections.abc import Callable

from langchain.schema import Document


### START contents of "[~]/biocypher/biocypher/_config/__init__.py" (with the aim of making the "from biocypher import _config" command below superfluous)
#
#
"""
Module data directory, including:

* The BioLink database schema
* The default config files
"""

import os
import warnings

from typing import Any, Optional

import appdirs
import yaml

__all__ = ["module_data", "module_data_path", "read_config", "config", "reset"]

_USER_CONFIG_DIR = appdirs.user_config_dir("biocypher", "saezlab")

_USER_CONFIG_FILE = os.path.join(_USER_CONFIG_DIR, "conf.yaml")


class MyLoader(yaml.SafeLoader):
    def construct_scalar(self, node):
        # Check if the scalar contains double quotes and an escape sequence
        value = super().construct_scalar(node)
        q = bool(node.style == '"')
        b = bool("\\" in value.encode("unicode_escape").decode("utf-8"))
        if q and b:
            warnings.warn(
                (
                    "Double quotes detected in YAML configuration scalar: "
                    f"{value.encode('unicode_escape')}. "
                    "These allow escape sequences and may cause problems, for "
                    "instance with the Neo4j admin import files (e.g. '\\t'). "
                    "Make sure you wanted to do this, and use single quotes "
                    "whenever possible."
                ),
                category=UserWarning,
            )
        return value


def module_data_path(name: str) -> str:
    """
    Absolute path to a YAML file shipped with the module.
    """

    here = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(here, f"{name}.yaml")


def module_data(name: str) -> Any:
    """
    Retrieve the contents of a YAML file shipped with this module.
    """

    path = module_data_path(name)

    return _read_yaml(path)


def _read_yaml(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r") as fp:
            return yaml.load(fp.read(), Loader=MyLoader)


def read_config() -> dict:
    """
    Read the module config.

    Read and merge the built-in default, the user level and directory level
    configuration, with the later taking precendence over the former.

    TODO explain path configuration
    """

    defaults = _read_yaml("[~]/biocypher/biocypher/_config/biocypher_config.yaml")

    user = _read_yaml(_USER_CONFIG_FILE) or {}
    # TODO account for .yml?
    #
    #
    local = _read_yaml("biocypher_config.yaml") or _read_yaml("config/biocypher_config.yaml") or {}

    for key in defaults:
        value = local[key] if key in local else user[key] if key in user else None

        if value is not None:
            if isinstance(defaults[key], str):  # first level config (like title)
                defaults[key] = value
            else:
                defaults[key].update(value)

    return defaults


def config(*args, **kwargs) -> Optional[Any]:
    """
    Set or get module config parameters.
    """

    if args and kwargs:
        raise ValueError(
            "Setting and getting values in the same call is not allowed.",
        )

    if args:
        result = tuple(globals()["_config"].get(key, None) for key in args)

        return result[0] if len(result) == 1 else result

    for key, value in kwargs.items():
        globals()["_config"][key].update(value)


def reset():
    """
    Reload configuration from the config files.
    """

    globals()["_config"] = read_config()


reset()


def update_from_file(path: str):
    """
    Update the module configuration from a YAML file.
    """

    config(**_read_yaml(path))
#
#
### END contents of "[~]/biocypher/biocypher/_config/__init__.py" (with the aim of making the "from biocypher import _config" command below superfluous)


### START contents of "[~]/biocypher/biocypher/_metadata.py" (with the aim of making the "from biocypher._metadata import __version__" command below superfluous)
#
#
"""
Package metadata (version, authors, etc).
"""

__all__ = ["get_metadata"]

import importlib.metadata
import os
import pathlib

try:
    import toml
except ImportError:
    toml = None

_VERSION = "0.12.0"


def get_metadata():
    """
    Basic package metadata.

    Retrieves package metadata from the current project directory or from
    the installed package.
    """

    here = pathlib.Path(__file__).parent
    pyproj_toml = "pyproject.toml"
    meta = {}

    for project_dir in (here, here.parent):
        toml_path = str(project_dir.joinpath(pyproj_toml).absolute())

        if os.path.exists(toml_path) and toml is not None:
            try:
                pyproject = toml.load(toml_path)
            except Exception:
                # If toml parsing fails, skip and use fallback
                continue

            # Use modern PEP 621 format (uv/hatchling)
            if "project" in pyproject:
                project = pyproject["project"]
                meta = {
                    "name": project.get("name"),
                    "version": project.get("version"),
                    "author": project.get("authors", []),
                    "license": project.get("license", {}).get("text"),
                    "full_metadata": pyproject,
                }
            elif "tool" in pyproject and "poetry" in pyproject["tool"]:
                # Legacy Poetry format fallback (for backward compatibility)
                poetry = pyproject["tool"]["poetry"]
                meta = {
                    "name": poetry.get("name"),
                    "version": poetry.get("version"),
                    "author": poetry.get("authors", []),
                    "license": poetry.get("license"),
                    "full_metadata": pyproject,
                }

            break

    if not meta:
        try:
            meta = {k.lower(): v for k, v in importlib.metadata.metadata(here.name).items()}

        except importlib.metadata.PackageNotFoundError:
            pass

    meta["version"] = meta.get("version", None) or _VERSION

    return meta


metadata = get_metadata()
__version__ = metadata.get("version", None)
__author__ = metadata.get("author", None)
__license__ = metadata.get("license", None)
#
#
### END contents of "[~]/biocypher/biocypher/_metadata.py" (with the aim of making the "from biocypher._metadata import __version__" command below superfluous)


### START contents of "/home/hammie/SOFTWARE/BioCypher_Chatter/biocypher/biocypher/_logger.py" (with the aim of making the "from biocypher._logger import logger" command below superfluous)
#
#
"""
Configuration of the module logger.
"""

__all__ = ["get_logger", "log", "logfile"]

import logging
import os
import pydoc

from datetime import datetime


def get_logger(name: str = "biocypher") -> logging.Logger:
    """
    Access the module logger, create a new one if does not exist yet.

    Method providing central logger instance to main module. Is called
    only from main submodule, :mod:`biocypher.driver`. In child modules,
    the standard Python logging facility is called
    (using ``logging.getLogger(__name__)``), automatically inheriting
    the handlers from the central logger.

    The file handler creates a log file named after the current date and
    time. Levels to output to file and console can be set here.

    Args:
        name:
            Name of the logger instance.

    Returns:
        An instance of the Python :py:mod:`logging.Logger`.
    """

    if not logging.getLogger(name).hasHandlers():
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

        # formatting
        file_formatter = logging.Formatter(
            "%(asctime)s\t%(levelname)s\tmodule:%(module)s\n%(message)s",
        )
        stdout_formatter = logging.Formatter("%(levelname)s -- %(message)s")

        # file name and creation
        now = datetime.now()
        date_time = now.strftime("%Y%m%d-%H%M%S")

        log_to_disk = config("biocypher").get("log_to_disk")

        if log_to_disk:

            logdir = config("biocypher").get("log_directory") or "biocypher-log"

            os.makedirs(logdir, exist_ok=True)
            logfile = os.path.join(logdir, f"biocypher-{date_time}.log")

            # file handler
            file_handler = logging.FileHandler(logfile)

            if config("biocypher").get("debug"):             
                file_handler.setLevel(logging.DEBUG)
            else:
                file_handler.setLevel(logging.INFO)

            file_handler.setFormatter(file_formatter)

            logger.addHandler(file_handler)

        # handlers
        # stream handler
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(stdout_formatter)

        # add handlers
        logger.addHandler(stdout_handler)

        # startup message
        logger.info(f"This is BioCypher v{__version__}.")
        if log_to_disk:
            logger.info(f"Logging into `{logfile}`.")
        else:
            logger.info("Logging into stdout.")

    return logging.getLogger(name)


def logfile() -> str:
    """
    Path to the log file.
    """

    return get_logger().handlers[0].baseFilename


def log():
    """
    Browse the log file.
    """

    with open(logfile()) as fp:
        pydoc.pager(fp.read())


logger = get_logger()
#
#
### END contents of "[~]/biocypher/biocypher/_logger.py".


### START contents of "[~]/biocypher/biocypher/_misc.py" (with the aim of making the "from biocypher._misc import to_list" command below superfluous)
#
#
"""Handy functions for use in various places."""

import re

from collections.abc import (
    Generator,
    ItemsView,
    Iterable,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import Any

import networkx as nx

from treelib import Tree

logger.debug(f"Loading module {__name__}.")

__all__ = ["LIST_LIKE", "SIMPLE_TYPES", "ensure_iterable", "to_list"]

SIMPLE_TYPES = (
    bytes,
    str,
    int,
    float,
    bool,
    type(None),
)

LIST_LIKE = (
    list,
    set,
    tuple,
    Generator,
    ItemsView,
    KeysView,
    Mapping,
    ValuesView,
)


def to_list(value: Any) -> list:
    """Ensure that ``value`` is a list."""
    if isinstance(value, LIST_LIKE):
        value = list(value)

    else:
        value = [value]

    return value


def ensure_iterable(value: Any) -> Iterable:
    """Return iterables, except strings, wrap simple types into tuple."""
    return value if isinstance(value, LIST_LIKE) else (value,)


def create_tree_visualisation(inheritance_graph: dict | nx.Graph) -> Tree:
    """Create a visualisation of the inheritance tree using treelib."""
    inheritance_tree = _get_inheritance_tree(inheritance_graph)
    classes, root = _find_root_node(inheritance_tree)

    tree = Tree()
    tree.create_node(root, root)
    while classes:
        for child in classes:
            parent = inheritance_tree[child]
            if parent in tree.nodes.keys() or parent == root:
                tree.create_node(child, child, parent=parent)

        for node in tree.nodes.keys():
            if node in classes:
                classes.remove(node)

    return tree


def _get_inheritance_tree(inheritance_graph: dict | nx.Graph) -> dict | None:
    """Transform an inheritance_graph into an inheritance_tree.

    Args:
    ----
        inheritance_graph: A dict or nx.Graph representing the inheritance graph.

    Returns:
    -------
        A dict representing the inheritance tree.

    """
    if isinstance(inheritance_graph, nx.Graph):
        inheritance_tree = nx.to_dict_of_lists(inheritance_graph)

        multiple_parents_present = _multiple_inheritance_present(inheritance_tree)
        if multiple_parents_present:
            msg = (
                "The ontology contains multiple inheritance (one child node "
                "has multiple parent nodes). This is not visualized in the "
                "following hierarchy tree (the child node is only added once). "
                "If you wish to browse all relationships of the parsed "
                "ontologies, write a graphml file to disk using "
                "`to_disk = <directory>` and view this file.",
            )
            logger.warning(msg)
        # unlist values
        inheritance_tree = {k: v[0] for k, v in inheritance_tree.items() if v}
        return inheritance_tree
    elif not _multiple_inheritance_present(inheritance_graph):
        return inheritance_graph
    return None  # Explicit return for the case when neither condition is met


def _multiple_inheritance_present(inheritance_tree: dict) -> bool:
    """Check if multiple inheritance is present in the inheritance_tree."""
    return any(len(value) > 1 for value in inheritance_tree.values())


def _find_root_node(inheritance_tree: dict) -> tuple[set, str]:
    classes = set(inheritance_tree.keys())
    parents = set(inheritance_tree.values())
    root = list(parents - classes)
    if len(root) > 1:
        if "entity" in root:
            root = "entity"  # TODO: default: good standard?
        else:
            msg = f"Inheritance tree cannot have more than one root node. Found {len(root)}: {root}."
            logger.error(msg)
            raise ValueError(msg)
    else:
        root = root[0]
    if not root:
        # find key whose value is None
        root = list(inheritance_tree.keys())[list(inheritance_tree.values()).index(None)]
    return classes, root


# string conversion, adapted from Biolink Model Toolkit
lowercase_pattern = re.compile(r"[a-zA-Z]*[a-z][a-zA-Z]*")
underscore_pattern = re.compile(r"(?<!^)(?=[A-Z][a-z])")


def from_pascal(s: str, sep: str = " ") -> str:
    underscored = underscore_pattern.sub(sep, s)
    lowercased = lowercase_pattern.sub(
        lambda match: match.group(0).lower(),
        underscored,
    )
    return lowercased


def pascalcase_to_sentencecase(s: str) -> str:
    """Convert PascalCase to sentence case.

    Args:
    ----
        s: Input string in PascalCase

    Returns:
    -------
        string in sentence case form

    """
    return from_pascal(s, sep=" ")


def snakecase_to_sentencecase(s: str) -> str:
    """Convert snake_case to sentence case.

    Args:
    ----
        s: Input string in snake_case

    Returns:
    -------
        string in sentence case form

    """
    return " ".join(word.lower() for word in s.split("_"))


def sentencecase_to_snakecase(s: str) -> str:
    """Convert sentence case to snake_case.

    Args:
    ----
        s: Input string in sentence case

    Returns:
    -------
        string in snake_case form

    """
    return "_".join(s.lower().split())


def sentencecase_to_pascalcase(s: str, sep: str = r"\s") -> str:
    """Convert sentence case to PascalCase.

    Args:
    ----
        s: Input string in sentence case
        sep: Separator for the words in the input string

    Returns:
    -------
        string in PascalCase form

    """
    return re.sub(
        r"(?:^|[" + sep + "])([a-zA-Z])",
        lambda match: match.group(1).upper(),
        s,
    )


def to_lower_sentence_case(s: str) -> str:
    """Convert any string to lower sentence case.

    Works with snake_case, PascalCase, and sentence case.

    Args:
    ----
        s: Input string

    Returns:
    -------
        string in lower sentence case form

    """
    if "_" in s:
        return snakecase_to_sentencecase(s)
    elif " " in s:
        return s.lower()
    elif s[0].isupper():
        return pascalcase_to_sentencecase(s)
    else:
        return s


def is_nested(lst: list) -> bool:
    """Check if a list is nested.

    Args:
    ----
        lst (list): The list to check.

    Returns:
    -------
        bool: True if the list is nested, False otherwise.

    """
    for item in lst:
        if isinstance(item, list):
            return True
    return False
#
#
### END contents of "[~]/biocypher/biocypher/_misc.py" (with the aim of making the "from biocypher._misc import to_list" command below superfluous).


### START contents of "[~]/biocypher/biocypher/output/connect/_neo4j_driver_wrapper.py" (as definition of "class Neo4jDriver")
#
#
"""
Neo4j connection management and Cypher interface.

A wrapper around the Neo4j driver which handles the DBMS connection and
provides basic management methods. This module is only used when BioCypher
is configured for online mode with Neo4j.
"""


import contextlib
import itertools
import os
import re
import warnings

from typing import Literal

import appdirs
import yaml

__all__ = ["CONFIG_FILES", "DEFAULT_CONFIG", "Neo4jDriver"]

# Try to import Neo4j driver, but don't fail if not available
try:
    import neo4j
    import neo4j.exceptions as neo4j_exc

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    neo4j = None
    neo4j_exc = None

CONFIG_FILES = Literal["neo4j.yaml", "neo4j.yml"]
DEFAULT_CONFIG = {
    "user": "neo4j",
    "passwd": "neo4j",
    "db": "neo4j",
    "uri": "neo4j://localhost:7687",
    "fetch_size": 1000,
    "raise_errors": False,
    "fallback_db": ("system", "neo4j"),
    "fallback_on": ("TransientError",),
}


def _to_tuple(value):
    """Ensure that value is a tuple."""
    return tuple(to_list(value))


def _to_set(value):
    """Ensure that value is a set."""
    return set(to_list(value))


def _if_none(*values):
    """Use the first item from values that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


def _pretty_profile(d, lines=None, indent=0):
    """
    Pretty format a Neo4j profile dict.

    Takes Neo4j profile dictionary and an optional header as
    list and creates a list of strings to be printed.

    Args:
        d: Profile dictionary or list
        lines: Optional list to append to
        indent: Indentation level

    Returns:
        List of formatted strings
    """
    if lines is None:
        lines = []

    # ANSI color codes for terminal output
    OKBLUE = "\033[94m"
    WARNING = "\033[93m"
    ENDC = "\033[0m"

    # if more items, branch
    if d:
        if isinstance(d, list):
            for sd in d:
                _pretty_profile(sd, lines, indent)
        elif isinstance(d, dict):
            typ = d.pop("operatorType", None)
            if typ:
                lines.append(("\t" * indent) + "|" + "\t" + f"{OKBLUE}Step: {typ} {ENDC}")

            # buffer children
            chi = d.pop("children", None)

            for key, value in d.items():
                if key == "args":
                    _pretty_profile(value, lines, indent)
                # both are there for some reason, sometimes
                # both in the same process
                elif key == "Time" or key == "time":
                    lines.append(
                        ("\t" * indent) + "|" + "\t" + str(key) + ": " + f"{WARNING}{value:,}{ENDC}".replace(",", " ")
                    )
                else:
                    lines.append(("\t" * indent) + "|" + "\t" + str(key) + ": " + str(value))

            # now the children
            _pretty_profile(chi, lines, indent + 1)

    return lines


def _get_neo4j_version(driver) -> str | None:
    """
    Get Neo4j version from the database.

    Args:
        driver: Neo4j driver instance

    Returns:
        Version string or None if unavailable
    """
    if not NEO4J_AVAILABLE or not driver:
        return None

    try:
        with driver.session() as session:
            result = session.run(
                """
                CALL dbms.components()
                YIELD name, versions, edition
                UNWIND versions AS version
                RETURN version AS version
                """
            )
            data = result.data()
            if data:
                return data[0]["version"]
    except Exception as e:
        logger.warning(f"Error detecting Neo4j version: {e}")
    return None


class Neo4jDriver:
    """
    Manage the connection to the Neo4j server.

    A wrapper around the Neo4j driver that handles database connections
    and provides convenient methods for querying and managing the database.
    """

    _connect_essential = ("uri", "user", "passwd")

    def __init__(
        self,
        driver: neo4j.Driver | Neo4jDriver | None = None,
        db_name: str | None = None,
        db_uri: str | None = None,
        db_user: str | None = None,
        db_passwd: str | None = None,
        config: CONFIG_FILES | None = None,
        fetch_size: int = 1000,
        raise_errors: bool | None = None,
        wipe: bool = False,
        offline: bool = False,
        fallback_db: str | tuple[str] | None = None,
        fallback_on: str | set[str] | None = None,
        multi_db: bool | None = None,
        force_enterprise: bool = False,
        querystr: str | None = None,
        **kwargs,
    ):
        """
        Create a Driver object with database connection and runtime parameters.

        Args:
            driver:
                A neo4j.Driver instance, created by neo4j.GraphDatabase.driver.
            db_name:
                Name of the database (Neo4j graph) to use.
            db_uri:
                Protocol, host and port to access the Neo4j server.
            db_user:
                Neo4j user name.
            db_passwd:
                Password of the Neo4j user.
            config:
                Path to a YAML config file which provides the URI, user
                name and password.
            fetch_size:
                Optional; the fetch size to use in database transactions.
            raise_errors:
                Raise the errors instead of turning them into log messages
                and returning None.
            wipe:
                Wipe the database after connection, ensuring the data is
                loaded into an empty database.
            offline:
                Disable any interaction to the server. Queries won't be
                executed. The config will be still stored in the object
                and it will be ready to go online by its go_online method.
            fallback_db:
                Arbitrary number of fallback databases. If a query fails
                to run against the current database, it will be attempted
                against the fallback databases.
            fallback_on:
                Switch to the fallback databases upon these errors.
            multi_db:
                Whether to use multi-database mode (Neo4j 4.0+).
            ### START text added based on "def _detect_and_handle_community_edition(self)" below:
            #
            force_enterprise:
                If Enterprise Edition is forced, skip detection.
            querystr:
            #
            ### END text added by HD as at 2026-01-04 22:20GMT.
            kwargs:
                Ignored.
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is not installed. Install it with: " "pip install neo4j>=5.0")

        self.driver = getattr(driver, "driver", driver)
        self._db_config = {
            "uri": db_uri,
            "user": db_user,
            "passwd": db_passwd,
            "db": db_name,
            "fetch_size": fetch_size,
            "raise_errors": raise_errors,
            "fallback_db": fallback_db,
            "fallback_on": fallback_on,
        }
        self._config_file = config
        self._drivers = {}
        self._queries = {}
        self._offline = offline
        self.multi_db = multi_db
        self._neo4j_version_cache = None
        self._force_enterprise = force_enterprise

        if self.driver:
            logger.info("Using the driver provided.")
            self._config_from_driver()
            self._register_current_driver()
        else:
            logger.info("No driver provided, initialising it from local config.")
            self.db_connect()

        # Detect Community Edition and adjust settings accordingly
        # Default to Community Edition (safer for CI) unless explicitly overridden
        self._detect_and_handle_community_edition()

        self.ensure_db()

        if wipe:
            self.wipe_db()

    def db_connect(self):
        """Connect to the database server."""
        if not self._connect_param_available:
            self.read_config()

        con_param = f"uri={self.uri}, auth=(user, ***)"
        logger.info(f"Attempting to connect: {con_param}")

        if self.offline:
            self.driver = None
            logger.info("Offline mode, not connecting to database.")
        else:
            self.driver = neo4j.GraphDatabase.driver(
                uri=self.uri,
                auth=self.auth,
            )
            logger.info("Opened database connection.")

        self._register_current_driver()

    def _detect_and_handle_community_edition(self):
        """
        Detect Community Edition and adjust settings for compatibility.

        Community Edition doesn't support multi-database, so we:
        1. Convert neo4j:// to bolt:// to avoid routing issues
        2. Disable multi_db mode
        3. Use default database 'neo4j' if a custom database was requested
        """
        if not self.driver or self.offline:
            return

        # If Enterprise Edition is forced, skip detection
        if self._force_enterprise:
            logger.info("Enterprise Edition mode forced. Skipping Community Edition detection.")
            return

        # Check if multi-database is supported (Enterprise Edition)
        # Use bolt:// for detection to avoid routing table issues
        original_uri = self.uri
        detection_uri = original_uri
        if original_uri.startswith("neo4j://"):
            detection_uri = original_uri.replace("neo4j://", "bolt://", 1)
        elif original_uri.startswith("neo4j+s://"):
            detection_uri = original_uri.replace("neo4j+s://", "bolt+s://", 1)

        # Create a temporary driver with bolt:// for detection
        temp_driver = None
        supports_multi_db = False
        try:
            temp_driver = neo4j.GraphDatabase.driver(uri=detection_uri, auth=self.auth)
            with temp_driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    CALL dbms.components()
                    YIELD edition
                    RETURN edition CONTAINS 'enterprise' AS is_enterprise
                    """
                )
                data = result.data()
                supports_multi_db = data[0].get("is_enterprise", False) if data else False
        except Exception as e:
            logger.debug(f"Error detecting Neo4j edition: {e}. Assuming Community Edition.")
            # If detection fails, assume Community Edition (safer)
            supports_multi_db = False
        finally:
            if temp_driver:
                temp_driver.close()

        # If Community Edition or detection failed, adjust settings
        if not supports_multi_db:
            logger.info(
                "Neo4j Community Edition detected (or detection failed). "
                "Multi-database features are not available. "
                "Adjusting configuration for compatibility."
            )

            # Convert neo4j:// to bolt:// to avoid routing table issues
            # (already converted for detection, but need to update main driver)
            try:
                if original_uri.startswith("neo4j://"):
                    bolt_uri = original_uri.replace("neo4j://", "bolt://", 1)
                    self._db_config["uri"] = bolt_uri
                    logger.info(f"Converted URI from {original_uri} to {bolt_uri} for Community Edition compatibility.")
                    # Reconnect with bolt://
                    self.driver.close()
                    self.db_connect()
                elif original_uri.startswith("neo4j+s://"):
                    bolt_uri = original_uri.replace("neo4j+s://", "bolt+s://", 1)
                    self._db_config["uri"] = bolt_uri
                    logger.info(f"Converted URI from {original_uri} to {bolt_uri} for Community Edition compatibility.")
                    # Reconnect with bolt+s://
                    self.driver.close()
                    self.db_connect()
            except Exception as e:
                logger.warning(f"Failed to convert URI and reconnect: {e}. Continuing with original URI.")

            # Disable multi_db mode
            if self.multi_db:
                logger.info("Disabling multi-database mode for Community Edition.")
                self.multi_db = False

            # Use default database if a custom database was requested
            current_db = self.current_db
            if current_db and current_db.lower() != "neo4j":
                logger.warning(
                    f"Requested database '{current_db}' is not supported in Community Edition. "
                    f"Falling back to default database 'neo4j'."
                )
                self._db_config["db"] = "neo4j"
                self._register_current_driver()

    @property
    def _connect_param_available(self) -> bool:
        """Check for essential connection parameters."""
        return all(self._db_config.get(k, None) for k in self._connect_essential)

    @property
    def status(
        self,
    ) -> Literal[
        "no driver",
        "no connection",
        "db offline",
        "db online",
        "offline",
    ]:
        """State of this driver object and its current database."""
        if self.offline:
            return "offline"

        if not self.driver:
            return "no driver"

        db_status = self.db_status()
        return f"db {db_status}" if db_status else "no connection"

    @property
    def uri(self) -> str:
        """Database server URI (from config or built-in default)."""
        return self._db_config.get("uri") or DEFAULT_CONFIG["uri"]

    @property
    def auth(self) -> tuple[str, str]:
        """Database server user and password (from config or built-in default)."""
        auth_tuple = self._db_config.get("auth")
        if auth_tuple:
            return tuple(auth_tuple)
        return (
            self._db_config.get("user") or DEFAULT_CONFIG["user"],
            self._db_config.get("passwd") or DEFAULT_CONFIG["passwd"],
        )

    def read_config(self, section: str | None = None):
        """Read the configuration from a YAML file."""
        config_key_synonyms = {
            "password": "passwd",
            "pw": "passwd",
            "username": "user",
            "login": "user",
            "host": "uri",
            "address": "uri",
            "server": "uri",
            "graph": "db",
            "database": "db",
            "name": "db",
        }

        if not self._config_file or not os.path.exists(self._config_file):
            confdirs = (".", appdirs.user_config_dir("biocypher", "biocypher"))
            conffiles = ("neo4j.yaml", "neo4j.yml")

            for config_path_t in itertools.product(confdirs, conffiles):
                config_path_s = os.path.join(*config_path_t)
                if os.path.exists(config_path_s):
                    self._config_file = config_path_s

        if self._config_file and os.path.exists(self._config_file):
            logger.info(f"Reading config from `{self._config_file}`.")

            with open(self._config_file) as fp:
                conf = yaml.safe_load(fp.read())

            for k, v in conf.get(section, conf).items():
                k = k.lower()
                k = config_key_synonyms.get(k, k)

                if not self._db_config.get(k, None):
                    self._db_config[k] = v

        elif not self._connect_param_available:
            logger.warning("No config available, falling back to defaults.")

        self._config_from_defaults()

    def _config_from_driver(self):
        """Extract configuration from an existing driver."""
        from_driver = {
            "uri": self._uri(
                host=getattr(self.driver, "default_host", None),
                port=getattr(self.driver, "default_port", None),
            ),
            "db": self.current_db,
            "fetch_size": getattr(
                getattr(self.driver, "_default_workspace_config", None),
                "fetch_size",
                None,
            ),
            "user": self.user,
            "passwd": self.passwd,
        }

        for k, v in from_driver.items():
            self._db_config[k] = self._db_config.get(k, v) or v

        self._config_from_defaults()

    def _config_from_defaults(self):
        """Populate missing config items by their default values."""
        for k, v in DEFAULT_CONFIG.items():
            if self._db_config.get(k, None) is None:
                self._db_config[k] = v

    def _register_current_driver(self):
        """Register the current driver for the current database."""
        self._drivers[self.current_db] = self.driver

    @staticmethod
    def _uri(
        host: str = "localhost",
        port: str | int = 7687,
        protocol: str = "neo4j",
    ) -> str:
        """Construct a Neo4j URI."""
        return f"{protocol}://{host}:{port}/"

    def close(self):
        """Close the Neo4j driver if it exists and is open."""
        if hasattr(self, "driver") and hasattr(self.driver, "close"):
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    @property
    def current_db(self) -> str:
        """Name of the current database."""
        return self._db_config["db"] or self._driver_con_db or self.home_db or neo4j.DEFAULT_DATABASE

    @current_db.setter
    def current_db(self, name: str):
        """Set the database currently in use."""
        self._db_config["db"] = name
        self.db_connect()

    @property
    def _driver_con_db(self) -> str | None:
        """Get the database from the driver connection."""
        if not self.driver:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                driver_con = self.driver.verify_connectivity()
            except neo4j_exc.ServiceUnavailable:
                logger.error("Cannot access Neo4j server.")
                return None

        if driver_con:
            first_con = next(iter(driver_con.values()))[0]
            return first_con.get("db", None)

        return None

    @property
    def home_db(self) -> str | None:
        """Home database of the current user."""
        return self._db_name("HOME")

    @property
    def default_db(self) -> str | None:
        """Default database of the server."""
        return self._db_name("DEFAULT")

    def _db_name(self, which: Literal["HOME", "DEFAULT"] = "HOME") -> str | None:
        """Get the HOME or DEFAULT database name."""
        try:
            resp, summary = self.query(
                f"SHOW {which} DATABASE;",
                fallback_db=self._get_fallback_db,
            )
        except (neo4j_exc.AuthError, neo4j_exc.ServiceUnavailable) as e:
            logger.error(f"No connection to Neo4j server: {e}")
            return None

        if resp:
            return resp[0]["name"]
        return None

    @property
    def _get_fallback_db(self) -> tuple[str]:
        """Get fallback database tuple."""
        return _to_tuple(getattr(self, "_fallback_db", None) or self._db_config["fallback_db"])

    @property
    def _get_fallback_on(self) -> set[str]:
        """Get fallback error types."""
        return _to_set(getattr(self, "_fallback_on", None) or self._db_config["fallback_on"])

    def query(
        self,
        querystr: str,
        db: str | None = None,
        fetch_size: int | None = None,
        write: bool = True,
        explain: bool = False,
        profile: bool = False,
        fallback_db: str | tuple[str] | None = None,
        fallback_on: str | set[str] | None = None,
        raise_errors: bool | None = None,
        parameters: dict | None = None,
        **kwargs,
    ) -> tuple[list[dict] | None, neo4j.work.summary.ResultSummary | None]:
        """
        Run a Cypher query.

        Args:
            querystr:
                A valid Cypher query.
            db:
                The DB inside the Neo4j server that should be queried.
            fetch_size:
                The Neo4j fetch size parameter.
            write:
                Indicates whether to address write- or read-servers.
            explain:
                Indicates whether to EXPLAIN the Cypher query.
            profile:
                Indicates whether to PROFILE the Cypher query.
            fallback_db:
                If the query fails, try to execute it against a fallback database.
            fallback_on:
                Run queries against the fallback databases in case of these errors.
            raise_errors:
                Raise Neo4j errors instead of only printing them.
            parameters:
                Parameters dictionary for the query.
            **kwargs:
                Additional parameters (deprecated, use parameters dict instead).

        Returns:
            2-tuple:
                - neo4j.Record.data: the Neo4j response to the query[(parametrised as 'querystr')]
                - neo4j.ResultSummary: information about the result
        """
        if explain:

            querystr = "EXPLAIN " + querystr

        elif profile:

            querystr = "PROFILE " + querystr

        if self.offline:

            logger.info(f"Offline mode, not running query: `{querystr}`.")

            return None, None

        if not self.driver:
            if raise_errors:
                raise RuntimeError("Driver is not available. The driver may be closed or in offline mode.")
            logger.error("Driver is not available. Cannot execute query.")
            return None, None

        # Check if driver is closed (Neo4j 5.x driver has _closed attribute)
        if hasattr(self.driver, "_closed") and self.driver._closed:
            if raise_errors:
                raise RuntimeError("Driver is closed. Please reconnect or create a new driver instance.")
            logger.error("Driver is closed. Cannot execute query.")
            return None, None

        db = db or self._db_config["db"] or neo4j.DEFAULT_DATABASE
        fetch_size = fetch_size or self._db_config["fetch_size"]
        raise_errors = self._db_config["raise_errors"] if raise_errors is None else raise_errors

        # Combine parameters dict with kwargs (kwargs for backward compatibility)
        query_params = dict(parameters or {}, **kwargs)

        # Neo4j 5+ uses database parameter, older versions use it conditionally
        session_kwargs = {
            "fetch_size": fetch_size,
            "default_access_mode": (neo4j.WRITE_ACCESS if write else neo4j.READ_ACCESS),
        }

        # For Neo4j 4.0+, use database parameter if multi_db is True
        # For Neo4j 5.0+, always use database parameter
        if self.multi_db or self._is_neo4j_5_plus():
            session_kwargs["database"] = db

        try:
            with self.session(**session_kwargs) as session:
                # Neo4j driver expects parameters via the 'parameters' argument,
                # not unpacked as kwargs. This ensures query parameters are correctly
                # passed to the Cypher query and prevents conflicts with method parameters.
                if query_params:

                    res = session.run(querystr, parameters=query_params)

                else:

                    res = session.run(querystr)

                return res.data(), res.consume()

        except (neo4j_exc.Neo4jError, neo4j_exc.DriverError) as e:
            fallback_db = fallback_db or getattr(self, "_fallback_db", ())
            fallback_on = _to_set(_if_none(fallback_on, self._get_fallback_on))

            if self._match_error(e, fallback_on):
                for fdb in _to_tuple(fallback_db):
                    if fdb != db:
                        logger.warning(f"Running query against fallback database `{fdb}`.")
                        return self.query(
                            querystr=querystr,
                            db=fdb,
                            fetch_size=fetch_size,
                            write=write,
                            fallback_on=set(),
                            raise_errors=raise_errors,
                            parameters=query_params,
                        )

            logger.error(f"Failed to run query: {e.__class__.__name__}: {e}")
            #
            logger.error(f"The error happened with this query: {querystr}")

            if e.__class__.__name__ == "AuthError":
                logger.error("Authentication error, switching to offline mode.")
                self.go_offline()

            if raise_errors:
                raise

            return None, None

    def _is_neo4j_5_plus(self) -> bool:
        """Check if Neo4j version is 5.0 or higher."""
        if self._neo4j_version_cache is None:
            version_str = _get_neo4j_version(self.driver)
            if version_str:
                try:
                    major_version = int(version_str.split(".")[0])
                    self._neo4j_version_cache = major_version >= 5
                except (ValueError, IndexError):
                    self._neo4j_version_cache = False
            else:
                self._neo4j_version_cache = False
        return self._neo4j_version_cache

    def explain(self, querystr, db=None, fetch_size=None, write=True, **kwargs):        
        """
        Explain a query and pretty print the output.

        Args:
            querystr: Cypher query to explain
            db: Database name
            fetch_size: Fetch size
            write: Write access mode
            **kwargs: Query parameters

        Returns:
            2-tuple:
                - dict: the raw plan returned by the Neo4j bolt driver
                - list of str: a list of strings ready for printing
        """
        logger.info("Explaining a query.")
        #
        data, summary = self.query(querystr, db, fetch_size, write, explain=True, **kwargs)

        if not summary:
            return None, []

        plan = summary.plan
        printout = _pretty_profile(plan)

        return plan, printout

    def profile(self, querystr, db=None, fetch_size=None, write=True, **kwargs):        
        """
        Profile a query and pretty print the output.

        Args:
            querystr: Cypher query to profile
            db: Database name
            fetch_size: Fetch size
            write: Write access mode
            **kwargs: Query parameters

        Returns:
            2-tuple:
                - dict: the raw profile returned by the Neo4j bolt driver
                - list of str: a list of strings ready for printing
        """
        logger.info("Profiling a query.")
        #
        data, summary = self.query(querystr, db, fetch_size, write, profile=True, **kwargs)

        if not summary:
            return None, []

        prof = summary.profile
        exec_time = summary.result_available_after + summary.result_consumed_after

        # get print representation
        header = f"Execution time: {exec_time:n}\n"
        printout = _pretty_profile(prof, [header], indent=0)

        return prof, printout

    def db_exists(self, name: str | None = None) -> bool:
        """Check if a database exists."""
        return bool(self.db_status(name=name))

    def db_status(
        self,
        name: str | None = None,
        field: str = "currentStatus",
    ) -> Literal["online", "offline"] | str | dict | None:
        """
        Get the current status or other state info of a database.

        Args:
            name: Name of a database
            field: The field to return

        Returns:
            The status as a string, None if the database does not exist.
            If field is None, a dictionary with all fields will be returned.
        """
        name = name or self.current_db
        #
        querystr = f'SHOW DATABASES WHERE name = "{name}";'

        # Use fallback context manager like original neo4j_utils
        # This allows query to default to current_db and fallback to system/neo4j on error
        with self.fallback():

            resp, summary = self.query(querystr)

        if resp:
            return resp[0].get(field, resp[0])
        return None

    def db_online(self, name: str | None = None) -> bool:
        """Check if a database is currently online."""
        return self.db_status(name=name) == "online"

    def create_db(self, name: str | None = None):
        """Create a database if it does not already exist."""
        self._manage_db("CREATE", name=name, options="IF NOT EXISTS")

    def start_db(self, name: str | None = None):
        """Start a database (bring it online) if it is offline."""
        self._manage_db("START", name=name)

    def stop_db(self, name: str | None = None):
        """Stop a database, making sure it's offline."""
        self._manage_db("STOP", name=name)

    def drop_db(self, name: str | None = None):
        """Delete a database if it exists."""
        self._manage_db("DROP", name=name, options="IF EXISTS")

    def _manage_db(
        self,
        cmd: Literal["CREATE", "START", "STOP", "DROP"],
        name: str | None = None,
        options: str | None = None,
    ):
        """Execute a database management command."""
        # Use fallback_db like original neo4j_utils
        # Query defaults to current_db, but fallback mechanism will retry against system/neo4j
        self.query(
            f"{cmd} DATABASE {name or self.current_db} {options or ''};",
            fallback_db=self._get_fallback_db,
        )

    def wipe_db(self):
        """Delete all contents of the current database."""
        if not self.driver:
            raise RuntimeError(
                "Driver is not available. Cannot wipe database. " "The driver may be closed or in offline mode."
            )

        # Check if driver is closed (Neo4j 5.x driver has _closed attribute)
        if hasattr(self.driver, "_closed") and self.driver._closed:
            raise RuntimeError(
                "Driver is closed. Cannot wipe database. " "Please reconnect or create a new driver instance."
            )

        # Ensure database exists before trying to wipe it
        self.ensure_db()

        # For Community Edition, use default database if current_db is not supported
        # Skip this check if Enterprise Edition is forced
        db_to_wipe = self.current_db
        if not self._force_enterprise:
            current_uri = self.uri
            is_neo4j_protocol = current_uri.startswith("neo4j://") or current_uri.startswith("neo4j+s://")
            is_non_default_db = db_to_wipe and db_to_wipe.lower() != "neo4j"
            is_community_edition = not self.multi_db or (is_neo4j_protocol and is_non_default_db)

            if is_community_edition and is_non_default_db:
                logger.warning(
                    f"Cannot wipe database '{db_to_wipe}' in Community Edition. "
                    f"Using default database 'neo4j' instead. "
                    f"Database will remain 'neo4j' for this session."
                )
                # Permanently change to default database for Community Edition
                db_to_wipe = "neo4j"
                self._db_config["db"] = "neo4j"
                self._register_current_driver()

        logger.info(f"Wiping database `{db_to_wipe}`.")
        self.query("MATCH (n) DETACH DELETE n;")
        self.drop_indices_constraints()

    def ensure_db(self):
        """Make sure the database exists and is online."""
        db_name = self.current_db

        # Skip if offline mode
        if self.offline:
            logger.debug(f"Offline mode, skipping database creation for '{db_name}'.")
            return

        # If Enterprise Edition is forced, skip Community Edition checks
        if self._force_enterprise:
            logger.debug(f"Enterprise Edition forced, proceeding with database check for '{db_name}'.")
            # Continue to database existence check below
        else:
            # In Community Edition, multi-database operations are not supported
            # The default database 'neo4j' always exists and is always online
            # Also skip if URI is bolt:// (which indicates Community Edition or direct connection)
            # If URI is still neo4j:// and we're checking a non-default database, assume Community Edition
            current_uri = self.uri
            is_bolt = current_uri.startswith("bolt://") or current_uri.startswith("bolt+s://")
            is_neo4j_protocol = current_uri.startswith("neo4j://") or current_uri.startswith("neo4j+s://")
            is_non_default_db = db_name and db_name.lower() != "neo4j"

            if not self.multi_db or is_bolt or (is_neo4j_protocol and is_non_default_db):
                if not self.multi_db:
                    logger.debug(
                        f"Multi-database mode disabled (Community Edition). "
                        f"Using default database '{db_name}' which always exists."
                    )
                elif is_bolt:
                    logger.debug(
                        f"Using bolt:// connection (direct mode). "
                        f"Using default database '{db_name}' which always exists."
                    )
                else:
                    logger.debug(
                        f"Using neo4j:// protocol with non-default database '{db_name}' - "
                        f"assuming Community Edition and skipping database check."
                    )
                return

        # Check if database exists, create if needed
        try:
            exists = self.db_exists()
            if not exists:
                logger.info(f"Database '{db_name}' does not exist, creating it...")
                self.create_db()
                # Verify creation succeeded
                if not self.db_exists():
                    raise RuntimeError(
                        f"Failed to create database '{db_name}'. " "The database was not created successfully."
                    )
                logger.info(f"Database '{db_name}' created successfully.")
            else:
                logger.debug(f"Database '{db_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to check/create database '{db_name}': {e}")
            # Re-raise to prevent initialization from continuing with a missing database
            raise RuntimeError(
                f"Failed to ensure database '{db_name}' exists: {e}. "
                "Please check Neo4j permissions and that the database can be created."
            ) from e

        # Check if database is online, start if needed
        try:
            if not self.db_online():
                logger.info(f"Database '{db_name}' is offline, starting it...")
                self.start_db()
                # Verify start succeeded
                if not self.db_online():
                    raise RuntimeError(
                        f"Failed to start database '{db_name}'. " "The database was not started successfully."
                    )
                logger.info(f"Database '{db_name}' started successfully.")
            else:
                logger.debug(f"Database '{db_name}' is already online.")
        except Exception as e:
            logger.error(f"Failed to check/start database '{db_name}': {e}")
            # Re-raise to prevent initialization from continuing with an offline database
            raise RuntimeError(
                f"Failed to ensure database '{db_name}' is online: {e}. "
                "Please check Neo4j permissions and that the database can be started."
            ) from e

    def select_db(self, name: str):
        """Set the current database."""
        current = self.current_db

        if current != name:
            self._register_current_driver()
            self._db_config["db"] = name

            if name in self._drivers:
                self.driver = self._drivers[name]
            else:
                self.db_connect()

    @property
    def indices(self) -> list | None:
        """List of indices in the current database."""
        return self._list_indices("indices")

    @property
    def constraints(self) -> list | None:
        """List of constraints in the current database."""
        return self._list_indices("constraints")

    def drop_indices_constraints(self):
        """Drop all indices and constraints in the current database."""
        # Neo4j 5+ handles constraints and indexes together
        self.drop_constraints()
        # For older versions, also drop indexes separately
        if not self._is_neo4j_5_plus():
            self.drop_indices()

    def drop_constraints(self):
        """Drop all constraints in the current database."""
        self._drop_indices(what="constraints")

    def drop_indices(self):
        """Drop all indices in the current database."""
        self._drop_indices(what="indexes")

    def _drop_indices(
        self,
        what: Literal["indexes", "indices", "constraints"] = "constraints",
    ):
        """Drop indices or constraints.

        Compatible with Neo4j 4.x and 5.x. Uses SHOW syntax which is
        available in both versions.
        """
        what_u = self._idx_cstr_synonyms(what)

        with self.session() as s:
            try:
                # SHOW INDEXES and SHOW CONSTRAINTS work in both Neo4j 4.x and 5.x
                # Neo4j 5.x unified constraints and indexes, but separate commands still work
                if what == "constraints":

                    querystr = "SHOW CONSTRAINTS"

                elif what in ("indexes", "indices"):

                    querystr = "SHOW INDEXES"

                else:

                    querystr = f"SHOW {what_u}S"  # Plural form

                indices = s.run(querystr)
                #
                #
                indices = list(indices)
                n_indices = len(indices)
                index_names = ", ".join(i["name"] for i in indices)

                for idx in indices:
                    s.run(f"DROP {what_u} `{idx['name']}` IF EXISTS")

                logger.info(f"Dropped {n_indices} {what}: {index_names}.")

            except (neo4j_exc.Neo4jError, neo4j_exc.DriverError) as e:
                logger.error(f"Failed to run query: {e}")

    def _list_indices(
        self,
        what: Literal["indexes", "indices", "constraints"] = "constraints",
    ) -> list | None:
        """List indices or constraints."""
        what_u = self._idx_cstr_synonyms(what)

        with self.session() as s:
            try:
                return list(s.run(f"SHOW {what_u.upper()};"))
            except (neo4j_exc.Neo4jError, neo4j_exc.DriverError) as e:
                logger.error(f"Failed to run query: {e}")
                return None

    @staticmethod
    def _idx_cstr_synonyms(what: str) -> str:
        """Convert index/constraint keyword to Cypher keyword."""
        what_s = {
            "indexes": "INDEX",
            "indices": "INDEX",
            "constraints": "CONSTRAINT",
        }

        what_u = what_s.get(what, None)

        if not what_u:
            msg = f'Allowed keywords are: "indexes", "indices" or "constraints", ' f"not `{what}`."
            logger.error(msg)
            raise ValueError(msg)

        return what_u

    @property
    def node_count(self) -> int | None:
        """Number of nodes in the database."""
        res, summary = self.query("MATCH (n) RETURN COUNT(n) AS count;")
        return res[0]["count"] if res else None

    @property
    def edge_count(self) -> int | None:
        """Number of edges in the database."""
        res, summary = self.query("MATCH ()-[r]->() RETURN COUNT(r) AS count;")
        return res[0]["count"] if res else None

    @property
    def user(self) -> str | None:
        """User for the currently active connection."""
        return self._extract_auth[0]

    @property
    def passwd(self) -> str | None:
        """Password for the currently active connection."""
        return self._extract_auth[1]

    @property
    def _extract_auth(self) -> tuple[str | None, str | None]:
        """Extract authentication data from the Neo4j driver."""
        auth = None, None

        if self.driver:
            opener_vars = self._opener_vars
            if "auth" in opener_vars:
                auth = opener_vars["auth"].cell_contents

        return auth

    @property
    def _opener_vars(self) -> dict:
        """Extract variables from the opener part of the Neo4j driver."""
        return dict(
            zip(
                self.driver._pool.opener.__code__.co_freevars,
                self.driver._pool.opener.__closure__,
            ),
        )

    def __len__(self):
        """Return the number of nodes in the database."""
        return self.node_count or 0

    @contextlib.contextmanager
    def use_db(self, name: str):
        """Context manager where the default database is set to name."""
        used_previously = self.current_db
        self.select_db(name=name)

        try:
            yield None
        finally:
            self.select_db(name=used_previously)

    @contextlib.contextmanager
    def fallback(
        self,
        db: str | tuple[str] | None = None,
        on: str | set[str] | None = None,
    ):
        """
        Context manager that attempts to run queries against a fallback database
        if running against the default database fails.
        """
        prev = {}

        for var in ("db", "on"):
            prev[var] = getattr(self, f"_fallback_{var}", None)
            setattr(
                self,
                f"_fallback_{var}",
                locals()[var] or self._db_config.get(f"fallback_{var}"),
            )

        try:
            yield None
        finally:
            for var in ("db", "on"):
                setattr(self, f"_fallback_{var}", prev[var])

    @contextlib.contextmanager
    def session(self, **kwargs):
        """Context manager with a database connection session."""
        if not self.driver:
            raise RuntimeError("Driver is not available. The driver may be closed or in offline mode.")

        # Check if driver is closed
        if hasattr(self.driver, "_closed") and self.driver._closed:
            raise RuntimeError("Driver is closed. Please reconnect or create a new driver instance.")

        session = self.driver.session(**kwargs)

        try:
            yield session
        finally:
            session.close()

    def __enter__(self):
        """Context manager entry."""
        self._context_session = self.session()
        return self._context_session

    def __exit__(self, *exc):
        """Context manager exit."""
        if hasattr(self, "_context_session"):
            self._context_session.close()
            delattr(self, "_context_session")

    def __repr__(self):
        """String representation."""
        return f"<{self.__class__.__name__} " f"{self._connection_str if self.driver else '[offline]'}>"

    @property
    def _connection_str(self) -> str:
        """Connection string representation."""
        if not self.driver:
            return "unknown://unknown:0/unknown"

        protocol = re.split(
            r"(?<=[a-z])(?=[A-Z])",
            self.driver.__class__.__name__,
        )[0].lower()

        address = self.driver._pool.address if hasattr(self.driver, "_pool") else ("unknown", 0)

        return f"{protocol}://{address[0]}:{address[1]}/{self.user or 'unknown'}"

    @property
    def offline(self) -> bool:
        """Whether the driver is in offline mode."""
        return self._offline

    @offline.setter
    def offline(self, offline: bool):
        """Enable or disable offline mode."""
        self.go_offline() if offline else self.go_online()

    @property
    def apoc_version(self) -> str | None:
        """
        Version of the APOC plugin available in the current database.

        Returns:
            APOC version string or None if APOC is not available
        """
        # Check if driver is available before attempting to query
        if not self.driver or self.offline:
            return None

        # Check if driver is closed
        if hasattr(self.driver, "_closed") and self.driver._closed:
            return None

        db = self._db_config["db"] or neo4j.DEFAULT_DATABASE

        try:
            with self.session(database=db) as session:
                res = session.run("RETURN apoc.version() AS output;")
                data = res.data()
                if data:
                    return data[0]["output"]
        except (neo4j_exc.ClientError, RuntimeError):
            # RuntimeError can be raised if driver is offline/closed
            # ClientError is raised if APOC is not available
            return None
        except Exception:
            # Catch any other exceptions (e.g., connection errors) and return None
            return None
        return None

    @property
    def has_apoc(self) -> bool:
        """
        Check if APOC is available in the current database.

        Returns:
            True if APOC is available, False otherwise
        """
        try:
            return bool(self.apoc_version)
        except Exception:
            # Ensure has_apoc always returns a boolean, even if apoc_version raises
            return False

    def go_offline(self):
        """Switch to offline mode."""
        self._offline = True
        self.close()
        self.driver = None
        self._register_current_driver()
        logger.warning("Offline mode: any interaction to the server is disabled.")

    def go_online(
        self,
        db_name: str | None = None,
        db_uri: str | None = None,
        db_user: str | None = None,
        db_passwd: str | None = None,
        config: CONFIG_FILES | None = None,
        fetch_size: int | None = None,
        raise_errors: bool | None = None,
        wipe: bool = False,
    ):
        """Switch to online mode."""
        self._offline = False

        try:
            for k, current in self._db_config.items():
                self._db_config[k] = _if_none(
                    locals().get(k.replace("db_", ""), None),
                    current,
                    DEFAULT_CONFIG.get(k),
                )

            self._config_file = self._config_file or config

            self.db_connect()
            self.ensure_db()
            logger.info("Online mode: ready to run queries.")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._offline = True

        if wipe:
            self.wipe_db()

    @staticmethod
    def _match_error(error: Exception | str, errors: set[Exception | str]) -> bool:
        """Check if error is listed in errors."""
        import builtins

        def str_to_exc(e):
            if isinstance(e, Exception):
                return e.__class__
            elif isinstance(e, str):
                return getattr(builtins, e, getattr(neo4j_exc, e, e))
            else:
                return e

        error = str_to_exc(error)
        errors = {str_to_exc(e) for e in _to_set(errors)}

        return error in errors or (
            isinstance(error, type) and any(issubclass(error, e) for e in errors if isinstance(e, type))
        )
#
#
### END contents of "[~]/biocypher/biocypher/output/connect/_neo4j_driver_wrapper.py".


### START contents of "[~]/biocypher/biocypher/_create.py" (with the aim of making the "from biocypher._create import BioCypherEdge, BioCypherNode" command below superfluous)
#
#
"""
BioCypher 'create' module. Handles the creation of BioCypher node and edge
dataclasses.
"""

import os

from dataclasses import dataclass, field
from typing import Union

logger.debug(f"Loading module {__name__}.")

__all__ = [
    "BioCypherEdge",
    "BioCypherNode",
    "BioCypherRelAsNode",
]


@dataclass(frozen=True)
class BioCypherNode:
    """
    Handoff class to represent biomedical entities as Neo4j nodes.

    Has id, label, property dict; id and label (in the Neo4j sense of a
    label, ie, the entity descriptor after the colon, such as
    ":Protein") are non-optional and called node_id and node_label to
    avoid confusion with "label" properties. Node labels are written in
    PascalCase and as nouns, as per Neo4j consensus.

    Args:
        node_id (string): consensus "best" id for biological entity
        node_label (string): primary type of entity, capitalised
        **properties (kwargs): collection of all other properties to be
            passed to neo4j for the respective node (dict)

    Todo:
        - check and correct small inconsistencies such as capitalisation
            of ID names ("uniprot" vs "UniProt")
        - check for correct ID patterns (eg "ENSG" + string of numbers,
            uniprot length)
        - ID conversion using pypath translation facilities for now
    """

    node_id: str
    node_label: str
    preferred_id: str = "id"
    properties: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Add id field to properties.

        Check for reserved keywords.

        Replace unwanted characters in properties.
        """
        self.properties["id"] = self.node_id
        self.properties["preferred_id"] = self.preferred_id or None
        # TODO actually make None possible here; as is, "id" is the default in
        # the dataclass as well as in the configuration file

        if ":TYPE" in self.properties.keys():
            logger.warning(
                "Keyword ':TYPE' is reserved for Neo4j. Removing from properties.",
                # "Renaming to 'type'."
            )
            # self.properties["type"] = self.properties[":TYPE"]
            del self.properties[":TYPE"]

        for k, v in self.properties.items():
            if isinstance(v, str):
                self.properties[k] = (
                    v.replace(
                        os.linesep,
                        " ",
                    )
                    .replace(
                        "\n",
                        " ",
                    )
                    .replace(
                        "\r",
                        " ",
                    )
                )

            elif isinstance(v, list):
                self.properties[k] = [
                    val.replace(
                        os.linesep,
                        " ",
                    )
                    .replace(
                        "\n",
                        " ",
                    )
                    .replace("\r", " ")
                    for val in v
                ]

    def get_id(self) -> str:
        """
        Returns primary node identifier.

        Returns:
            str: node_id
        """
        return self.node_id

    def get_label(self) -> str:
        """
        Returns primary node label.

        Returns:
            str: node_label
        """
        return self.node_label

    def get_type(self) -> str:
        """
        Returns primary node label.

        Returns:
            str: node_label
        """
        return self.node_label

    def get_preferred_id(self) -> str:
        """
        Returns preferred id.

        Returns:
            str: preferred_id
        """
        return self.preferred_id

    def get_properties(self) -> dict:
        """
        Returns all other node properties apart from primary id and
        label as key-value pairs.

        Returns:
            dict: properties
        """
        return self.properties

    def get_dict(self) -> dict:
        """
        Return dict of id, labels, and properties.

        Returns:
            dict: node_id and node_label as top-level key-value pairs,
            properties as second-level dict.
        """
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class BioCypherEdge:
    """
    Handoff class to represent biomedical relationships in Neo4j.

    Has source and target ids, label, property dict; ids and label (in
    the Neo4j sense of a label, ie, the entity descriptor after the
    colon, such as ":TARGETS") are non-optional and called source_id,
    target_id, and relationship_label to avoid confusion with properties
    called "label", which usually denotes the human-readable form.
    Relationship labels are written in UPPERCASE and as verbs, as per
    Neo4j consensus.

    Args:

        source_id (string): consensus "best" id for biological entity

        target_id (string): consensus "best" id for biological entity

        relationship_label (string): type of interaction, UPPERCASE

        properties (dict): collection of all other properties of the
        respective edge

    """

    source_id: str
    target_id: str
    relationship_label: str
    relationship_id: str = None
    properties: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Check for reserved keywords.
        """

        if ":TYPE" in self.properties.keys():
            logger.debug(
                "Keyword ':TYPE' is reserved for Neo4j. Removing from properties.",
                # "Renaming to 'type'."
            )
            # self.properties["type"] = self.properties[":TYPE"]
            del self.properties[":TYPE"]
        elif "id" in self.properties.keys():
            logger.debug(
                "Keyword 'id' is reserved for Neo4j. Removing from properties.",
                # "Renaming to 'type'."
            )
            # self.properties["type"] = self.properties[":TYPE"]
            del self.properties["id"]
        elif "_ID" in self.properties.keys():
            logger.debug(
                "Keyword '_ID' is reserved for Postgres. Removing from properties.",
                # "Renaming to 'type'."
            )
            # self.properties["type"] = self.properties[":TYPE"]
            del self.properties["_ID"]

    def get_id(self) -> Union[str, None]:
        """
        Returns primary node identifier or None.

        Returns:
            str: node_id
        """

        return self.relationship_id

    def get_source_id(self) -> str:
        """
        Returns primary node identifier of relationship source.

        Returns:
            str: source_id
        """
        return self.source_id

    def get_target_id(self) -> str:
        """
        Returns primary node identifier of relationship target.

        Returns:
            str: target_id
        """
        return self.target_id

    def get_label(self) -> str:
        """
        Returns relationship label.

        Returns:
            str: relationship_label
        """
        return self.relationship_label

    def get_type(self) -> str:
        """
        Returns relationship label.

        Returns:
            str: relationship_label
        """
        return self.relationship_label

    def get_properties(self) -> dict:
        """
        Returns all other relationship properties apart from primary ids
        and label as key-value pairs.

        Returns:
            dict: properties
        """
        return self.properties

    def get_dict(self) -> dict:
        """
        Return dict of ids, label, and properties.

        Returns:
            dict: source_id, target_id and relationship_label as
                top-level key-value pairs, properties as second-level
                dict.
        """
        return {
            "relationship_id": self.relationship_id or None,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_label": self.relationship_label,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class BioCypherRelAsNode:
    """
    Class to represent relationships as nodes (with in- and outgoing
    edges) as a triplet of a BioCypherNode and two BioCypherEdges. Main
    usage in type checking (instances where the receiving function needs
    to check whether it receives a relationship as a single edge or as
    a triplet).

    Args:

        node (BioCypherNode): node representing the relationship

        source_edge (BioCypherEdge): edge representing the source of the
            relationship

        target_edge (BioCypherEdge): edge representing the target of the
            relationship

    """

    node: BioCypherNode
    source_edge: BioCypherEdge
    target_edge: BioCypherEdge

    def __post_init__(self):
        if not isinstance(self.node, BioCypherNode):
            raise TypeError(
                f"BioCypherRelAsNode.node must be a BioCypherNode, " f"not {type(self.node)}.",
            )

        if not isinstance(self.source_edge, BioCypherEdge):
            raise TypeError(
                f"BioCypherRelAsNode.source_edge must be a BioCypherEdge, " f"not {type(self.source_edge)}.",
            )

        if not isinstance(self.target_edge, BioCypherEdge):
            raise TypeError(
                f"BioCypherRelAsNode.target_edge must be a BioCypherEdge, " f"not {type(self.target_edge)}.",
            )

    def get_node(self) -> BioCypherNode:
        return self.node

    def get_source_edge(self) -> BioCypherEdge:
        return self.source_edge

    def get_target_edge(self) -> BioCypherEdge:
        return self.target_edge
#
#
### END contents of "[~]/biocypher/biocypher/_create.py" (with the aim of making the "from biocypher._create import BioCypherEdge, BioCypherNode" command below superfluous).


### START contents of "[~]/biocypher/biocypher/_mapping.py" (with the aim of making the "from ._mapping import OntologyMapping" command below superfluous)
#
#
"""
BioCypher 'mapping' module. Handles the mapping of user-defined schema to the
underlying ontology.
"""

from typing import Optional
from urllib.request import urlopen

import yaml

logger.debug(f"Loading module {__name__}.")


class OntologyMapping:
    """
    Class to store the ontology mapping and extensions.
    """

    def __init__(self, config_file: str = None):
        self.schema = self._read_config(config_file)

        self.extended_schema = self._extend_schema()

    def _read_config(self, config_file: str = None):
        """
        Read the configuration file and store the ontology mapping and extensions.
        """
        if config_file is None:
            schema_config = {}

        # load yaml file from web
        elif config_file.startswith("http"):
            with urlopen(config_file) as f:
                schema_config = yaml.safe_load(f)

        # get graph state from config (assume file is local)
        else:
            with open(config_file, "r") as f:
                schema_config = yaml.safe_load(f)

        return schema_config

    def _extend_schema(self, d: Optional[dict] = None) -> dict:
        """
        Get leaves of the tree hierarchy from the data structure dict
        contained in the `schema_config.yaml`. Creates virtual leaves
        (as children) from entries that provide more than one preferred
        id type (and corresponding inputs).

        Args:
            d:
                Data structure dict from yaml file.

        """

        d = d or self.schema

        extended_schema = dict()

        # first pass: get parent leaves with direct representation in ontology
        for k, v in d.items():
            # k is not an entity
            if "represented_as" not in v:
                continue

            # preferred_id optional: if not provided, use `id`
            if not v.get("preferred_id"):
                v["preferred_id"] = "id"

            # k is an entity that is present in the ontology
            if "is_a" not in v:
                extended_schema[k] = v

        # second pass: "vertical" inheritance
        d = self._vertical_property_inheritance(d)
        for k, v in d.items():
            if "is_a" in v:
                # prevent loops
                if k == v["is_a"]:
                    logger.warning(
                        f"Loop detected in ontology mapping: {k} -> {v}. "
                        "Removing item. Please fix the inheritance if you want "
                        "to use this item."
                    )
                    continue

                extended_schema[k] = v

        # "horizontal" inheritance: create siblings for multiple identifiers or
        # sources -> virtual leaves or implicit children
        mi_leaves = {}
        ms_leaves = {}
        for k, v in d.items():
            # k is not an entity
            if "represented_as" not in v:
                continue

            if isinstance(v.get("preferred_id"), list):
                mi_leaves = self._horizontal_inheritance_pid(k, v)
                extended_schema.update(mi_leaves)

            elif isinstance(v.get("source"), list):
                ms_leaves = self._horizontal_inheritance_source(k, v)
                extended_schema.update(ms_leaves)

        return extended_schema

    def _vertical_property_inheritance(self, d):
        """
        Inherit properties from parents to children and update `d` accordingly.
        """
        for k, v in d.items():
            # k is not an entity
            if "represented_as" not in v:
                continue

            # k is an entity that is present in the ontology
            if "is_a" not in v:
                continue

            # "vertical" inheritance: inherit properties from parent
            if v.get("inherit_properties", False):
                # get direct ancestor
                if isinstance(v["is_a"], list):
                    parent = v["is_a"][0]
                else:
                    parent = v["is_a"]

                # ensure child has properties and exclude_properties
                if "properties" not in v:
                    v["properties"] = {}
                if "exclude_properties" not in v:
                    v["exclude_properties"] = {}

                # update properties of child
                parent_props = self.schema[parent].get("properties", {})
                if parent_props:
                    v["properties"].update(parent_props)

                parent_excl_props = self.schema[parent].get("exclude_properties", {})
                if parent_excl_props:
                    v["exclude_properties"].update(parent_excl_props)

                # update schema (d)
                d[k] = v

        return d

    def _horizontal_inheritance_pid(self, key, value):
        """
        Create virtual leaves for multiple preferred id types or sources.

        If we create virtual leaves, input_label/label_in_input always has to be
        a list.
        """

        leaves = {}

        preferred_id = value["preferred_id"]
        input_label = value.get("input_label") or value["label_in_input"]
        represented_as = value["represented_as"]

        # adjust lengths
        max_l = max(
            [
                len(_misc.to_list(preferred_id)),
                len(_misc.to_list(input_label)),
                len(_misc.to_list(represented_as)),
            ],
        )

        # adjust pid length if necessary
        if isinstance(preferred_id, str):
            pids = [preferred_id] * max_l
        else:
            pids = preferred_id

        # adjust rep length if necessary
        if isinstance(represented_as, str):
            reps = [represented_as] * max_l
        else:
            reps = represented_as

        for pid, lab, rep in zip(pids, input_label, reps):
            skey = pid + "." + key
            svalue = {
                "preferred_id": pid,
                "input_label": lab,
                "represented_as": rep,
                # mark as virtual
                "virtual": True,
            }

            # inherit is_a if exists
            if "is_a" in value.keys():
                # treat as multiple inheritance
                if isinstance(value["is_a"], list):
                    v = list(value["is_a"])
                    v.insert(0, key)
                    svalue["is_a"] = v

                else:
                    svalue["is_a"] = [key, value["is_a"]]

            else:
                # set parent as is_a
                svalue["is_a"] = key

            # inherit everything except core attributes
            for k, v in value.items():
                if k not in [
                    "is_a",
                    "preferred_id",
                    "input_label",
                    "label_in_input",
                    "represented_as",
                ]:
                    svalue[k] = v

            leaves[skey] = svalue

        return leaves

    def _horizontal_inheritance_source(self, key, value):
        """
        Create virtual leaves for multiple sources.

        If we create virtual leaves, input_label/label_in_input always has to be
        a list.
        """

        leaves = {}

        source = value["source"]
        input_label = value.get("input_label") or value["label_in_input"]
        represented_as = value["represented_as"]

        # adjust lengths
        src_l = len(source)

        # adjust label length if necessary
        if isinstance(input_label, str):
            labels = [input_label] * src_l
        else:
            labels = input_label

        # adjust rep length if necessary
        if isinstance(represented_as, str):
            reps = [represented_as] * src_l
        else:
            reps = represented_as

        for src, lab, rep in zip(source, labels, reps):
            skey = src + "." + key
            svalue = {
                "source": src,
                "input_label": lab,
                "represented_as": rep,
                # mark as virtual
                "virtual": True,
            }

            # inherit is_a if exists
            if "is_a" in value.keys():
                # treat as multiple inheritance
                if isinstance(value["is_a"], list):
                    v = list(value["is_a"])
                    v.insert(0, key)
                    svalue["is_a"] = v

                else:
                    svalue["is_a"] = [key, value["is_a"]]

            else:
                # set parent as is_a
                svalue["is_a"] = key

            # inherit everything except core attributes
            for k, v in value.items():
                if k not in [
                    "is_a",
                    "source",
                    "input_label",
                    "label_in_input",
                    "represented_as",
                ]:
                    svalue[k] = v

            leaves[skey] = svalue

        return leaves
#
#
### END contents of "[~]/biocypher/biocypher/_mapping.py" (with the aim of making the "from ._mapping import OntologyMapping" command below superfluous).


### START contents of "[~]/biocypher/biocypher/_ontology.py" (with the aim of making the "from ._ontology import Ontology" command below superfluous)
#
#
"""BioCypher 'ontology' module to parse and represent ontologies.

Also performs ontology hybridisation and other advanced operations.
"""

import os

from datetime import datetime
from itertools import chain
from typing import Optional

import networkx as nx
import rdflib

from rdflib import Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph

logger.debug(f"Loading module {__name__}.")


class OntologyAdapter:
    """Class that represents an ontology to be used in the Biocypher framework.

    Can read from a variety of formats, including OWL, OBO, and RDF/XML. The
    ontology is represented by a networkx.DiGraph object; an RDFlib graph is
    also kept. By default, the DiGraph reverses the label and identifier of the
    nodes, such that the node name in the graph is the human-readable label. The
    edges are oriented from child to parent. Labels are formatted in lower
    sentence case and underscores are replaced by spaces. Identifiers are taken
    as defined and the prefixes are removed by default.
    """

    def __init__(
        self,
        ontology_file: str,
        root_label: str,
        ontology_file_format: str | None = None,
        head_join_node_label: str | None = None,
        merge_nodes: bool | None = True,
        switch_label_and_id: bool = True,
        remove_prefixes: bool = True,
    ):
        """Initialize the OntologyAdapter class.

        Args:
        ----
            ontology_file (str): Path to the ontology file. Can be local or
                remote.

            root_label (str): The label of the root node in the ontology. In
                case of a tail ontology, this is the tail join node.

            ontology_file_format (str): The format of the ontology file (e.g. "application/rdf+xml")
                If format is not passed, it is determined automatically.

            head_join_node_label (str): Optional variable to store the label of the
                node in the head ontology that should be used to join to the
                root node of the tail ontology. Defaults to None.

            merge_nodes (bool): If True, head and tail join nodes will be
                merged, using the label of the head join node. If False, the
                tail join node will be attached as a child of the head join
                node.

            switch_label_and_id (bool): If True, the node names in the graph will be
                the human-readable labels. If False, the node names will be the
                identifiers. Defaults to True.

            remove_prefixes (bool): If True, the prefixes of the identifiers will
                be removed. Defaults to True.

        """
        logger.info(f"Instantiating OntologyAdapter class for {ontology_file}.")

        self._ontology_file = ontology_file
        self._root_label = root_label
        self._format = ontology_file_format
        self._merge_nodes = merge_nodes
        self._head_join_node = head_join_node_label
        self._switch_label_and_id = switch_label_and_id
        self._remove_prefixes = remove_prefixes

        self._rdf_graph = self._load_rdf_graph(ontology_file)

        self._nx_graph = self._rdf_to_nx(self._rdf_graph, root_label, switch_label_and_id)

    def _rdf_to_nx(
        self,
        _rdf_graph: rdflib.Graph,
        root_label: str,
        switch_label_and_id: bool,
        rename_nodes: bool = True,
    ) -> nx.DiGraph:
        one_to_one_triples, one_to_many_dict = self._get_relevant_rdf_triples(_rdf_graph)
        nx_graph = self._convert_to_nx(one_to_one_triples, one_to_many_dict)
        nx_graph = self._add_labels_to_nodes(nx_graph, switch_label_and_id)
        nx_graph = self._change_nodes_to_biocypher_format(nx_graph, switch_label_and_id, rename_nodes)
        nx_graph = self._get_all_ancestors(nx_graph, root_label, switch_label_and_id, rename_nodes)
        return nx.DiGraph(nx_graph)

    def _get_relevant_rdf_triples(self, g: rdflib.Graph) -> tuple:
        one_to_one_inheritance_graph = self._get_one_to_one_inheritance_triples(g)
        intersection = self._get_multiple_inheritance_dict(g)
        return one_to_one_inheritance_graph, intersection

    def _get_one_to_one_inheritance_triples(self, g: rdflib.Graph) -> rdflib.Graph:
        """Get the one to one inheritance triples from the RDF graph.

        Args:
        ----
            g (rdflib.Graph): The RDF graph

        Returns:
        -------
            rdflib.Graph: The one to one inheritance graph

        """
        one_to_one_inheritance_graph = Graph()
        # for s, p, o in g.triples((None, rdflib.RDFS.subClassOf, None)):
        for s, p, o in chain(
            g.triples((None, rdflib.RDFS.subClassOf, None)),  # Node classes
            g.triples((None, rdflib.RDF.type, rdflib.RDFS.Class)),  # Root classes
            g.triples((None, rdflib.RDFS.subPropertyOf, None)),  # OWL "edges" classes
            g.triples((None, rdflib.RDF.type, rdflib.OWL.ObjectProperty)),  # OWL "edges" root classes
        ):
            if self.has_label(s, g):
                one_to_one_inheritance_graph.add((s, p, o))
        return one_to_one_inheritance_graph

    def _get_multiple_inheritance_dict(self, g: rdflib.Graph) -> dict:
        """Get the multiple inheritance dictionary from the RDF graph.

        Args:
        ----
            g (rdflib.Graph): The RDF graph

        Returns:
        -------
            dict: The multiple inheritance dictionary

        """
        multiple_inheritance = g.triples((None, rdflib.OWL.intersectionOf, None))
        intersection = {}
        for (
            node,
            has_multiple_parents,
            first_node_of_intersection_list,
        ) in multiple_inheritance:
            parents = self._retrieve_rdf_linked_list(first_node_of_intersection_list)
            child_name = None
            for s_, _, _ in chain(
                g.triples((None, rdflib.RDFS.subClassOf, node)),
                g.triples((None, rdflib.RDFS.subPropertyOf, node)),
            ):
                child_name = s_

            # Handle Snomed CT post coordinated expressions
            if not child_name:
                for s_, _, _ in g.triples((None, rdflib.OWL.equivalentClass, node)):
                    child_name = s_

            if child_name:
                intersection[node] = {
                    "child_name": child_name,
                    "parent_node_names": parents,
                }
        return intersection

    def has_label(self, node: rdflib.URIRef, g: rdflib.Graph) -> bool:
        """Check if the node has a label in the graph.

        Args:
        ----
            node (rdflib.URIRef): The node to check
            g (rdflib.Graph): The graph to check in
        Returns:
            bool: True if the node has a label, False otherwise

        """
        return (node, rdflib.RDFS.label, None) in g

    def _retrieve_rdf_linked_list(self, subject: rdflib.URIRef) -> list:
        """Recursively retrieve a linked list from RDF.

        Example RDF list with the items [item1, item2]:
        list_node - first -> item1
        list_node - rest -> list_node2
        list_node2 - first -> item2
        list_node2 - rest -> nil

        Args:
        ----
            subject (rdflib.URIRef): One list_node of the RDF list

        Returns:
        -------
            list: The items of the RDF list

        """
        g = self._rdf_graph
        rdf_list = []
        for s, p, o in g.triples((subject, rdflib.RDF.first, None)):
            rdf_list.append(o)
        for s, p, o in g.triples((subject, rdflib.RDF.rest, None)):
            if o != rdflib.RDF.nil:
                rdf_list.extend(self._retrieve_rdf_linked_list(o))
        return rdf_list

    def _convert_to_nx(self, one_to_one: rdflib.Graph, one_to_many: dict) -> nx.DiGraph:
        """Convert the one to one and one to many inheritance graphs to networkx.

        Args:
        ----
            one_to_one (rdflib.Graph): The one to one inheritance graph
            one_to_many (dict): The one to many inheritance dictionary

        Returns:
        -------
            nx.DiGraph: The networkx graph

        """
        nx_graph = rdflib_to_networkx_digraph(one_to_one, edge_attrs=lambda s, p, o: {}, calc_weights=False)
        for key, value in one_to_many.items():
            nx_graph.add_edges_from([(value["child_name"], parent) for parent in value["parent_node_names"]])
            if key in nx_graph.nodes:
                nx_graph.remove_node(key)
        return nx_graph

    def _add_labels_to_nodes(self, nx_graph: nx.DiGraph, switch_label_and_id: bool) -> nx.DiGraph:
        """Add labels to the nodes in the networkx graph.

        Args:
        ----
            nx_graph (nx.DiGraph): The networkx graph
            switch_label_and_id (bool): If True, id and label are switched

        Returns:
        -------
            nx.DiGraph: The networkx graph with labels

        """
        for node in list(nx_graph.nodes):
            nx_id, nx_label = self._get_nx_id_and_label(node, switch_label_and_id)
            if nx_id == "none":
                # remove node if it has no id
                nx_graph.remove_node(node)
                continue

            nx_graph.nodes[node]["label"] = nx_label
        return nx_graph

    def _change_nodes_to_biocypher_format(
        self,
        nx_graph: nx.DiGraph,
        switch_label_and_id: bool,
        rename_nodes: bool = True,
    ) -> nx.DiGraph:
        """Change the nodes in the networkx graph to BioCypher format.

        This involves:
            - removing the prefix of the identifier
            - switching the id and label if requested
            - adapting the labels (replace _ with space and convert to lower
                sentence case)
        Args:
        ----
            nx_graph (nx.DiGraph): The networkx graph
            switch_label_and_id (bool): If True, id and label are switched
            rename_nodes (bool): If True, the nodes are renamed

        Returns:
        -------
            nx.DiGraph: The networkx ontology graph in BioCypher format

        """
        mapping = {
            node: self._get_nx_id_and_label(node, switch_label_and_id, rename_nodes)[0] for node in nx_graph.nodes
        }
        renamed = nx.relabel_nodes(nx_graph, mapping, copy=False)
        return renamed

    def _get_all_ancestors(
        self,
        renamed: nx.DiGraph,
        root_label: str,
        switch_label_and_id: bool,
        rename_nodes: bool = True,
    ) -> nx.DiGraph:
        """Get all ancestors of the root node in the networkx graph.

        Args:
        ----
            renamed (nx.DiGraph): The renamed networkx graph
            root_label (str): The label of the root node in the ontology
            switch_label_and_id (bool): If True, id and label are switched
            rename_nodes (bool): If True, the nodes are renamed

        Returns:
        -------
            nx.DiGraph: The filtered networkx graph

        """
        root = self._get_nx_id_and_label(
            self._find_root_label(self._rdf_graph, root_label),
            switch_label_and_id,
            rename_nodes,
        )[0]
        ancestors = nx.ancestors(renamed, root)
        ancestors.add(root)
        filtered_graph = renamed.subgraph(ancestors)
        return filtered_graph

    def _get_nx_id_and_label(self, node, switch_id_and_label: bool, rename_nodes: bool = True) -> tuple[str, str]:
        """Rename node id and label for nx graph.

        Args:
        ----
            node (str): The node to rename
            switch_id_and_label (bool): If True, switch id and label

        Returns:
        -------
            tuple[str, str]: The renamed node id and label

        """
        node_id_str = self._remove_prefix(str(node))
        node_label_str = str(self._rdf_graph.value(node, rdflib.RDFS.label))
        if rename_nodes:
            node_label_str = node_label_str.replace("_", " ")
            node_label_str = to_lower_sentence_case(node_label_str)
        nx_id = node_label_str if switch_id_and_label else node_id_str
        nx_label = node_id_str if switch_id_and_label else node_label_str
        return nx_id, nx_label

    def _find_root_label(self, g, root_label):
        # Loop through all labels in the ontology
        for label_subject, _, label_in_ontology in g.triples((None, rdflib.RDFS.label, None)):
            # If the label is the root label, set the root node to the label's subject
            if str(label_in_ontology) == root_label:
                root = label_subject
                break
        else:
            labels_in_ontology = []
            for label_subject, _, label_in_ontology in g.triples((None, rdflib.RDFS.label, None)):
                labels_in_ontology.append(str(label_in_ontology))
            msg = (
                f"Could not find root node with label '{root_label}'. "
                f"The ontology contains the following labels: {labels_in_ontology}"
            )
            logger.error(msg)
            raise ValueError(msg)
        return root

    def _remove_prefix(self, uri: str) -> str:
        """Remove the prefix of a URI.

        URIs can contain either "#" or "/" as a separator between the prefix
        and the local name. The prefix is everything before the last separator.

        Args:
        ----
            uri (str): The URI to remove the prefix from

        Returns:
        -------
            str: The URI without the prefix

        """
        if self._remove_prefixes:
            return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        else:
            return uri

    def _load_rdf_graph(self, ontology_file):
        """Load the ontology into an RDFlib graph.

        The ontology file can be in OWL, OBO, or RDF/XML format.

        Args:
        ----
            ontology_file (str): The path to the ontology file

        Returns:
        -------
            rdflib.Graph: The RDFlib graph

        """
        g = rdflib.Graph()
        g.parse(ontology_file, format=self._get_format(ontology_file))
        return g

    def _get_format(self, ontology_file):
        """Get the format of the ontology file."""
        if self._format:
            if self._format == "owl":
                return "application/rdf+xml"
            elif self._format == "obo":
                raise NotImplementedError("OBO format not yet supported")
            elif self._format == "rdf":
                return "application/rdf+xml"
            elif self._format == "ttl":
                return self._format
            else:
                msg = f"Could not determine format of ontology file {ontology_file}"
                logger.error(msg)
                raise ValueError(msg)

        if ontology_file.endswith(".owl"):
            return "application/rdf+xml"
        elif ontology_file.endswith(".obo"):
            msg = "OBO format not yet supported"
            logger.error(msg)
            raise NotImplementedError(msg)
        elif ontology_file.endswith(".rdf"):
            return "application/rdf+xml"
        elif ontology_file.endswith(".ttl"):
            return "ttl"
        else:
            msg = f"Could not determine format of ontology file {ontology_file}"
            logger.error(msg)
            raise ValueError(msg)

    def get_nx_graph(self):
        """Get the networkx graph representing the ontology."""
        return self._nx_graph

    def get_rdf_graph(self):
        """Get the RDFlib graph representing the ontology."""
        return self._rdf_graph

    def get_root_node(self):
        """Get root node in the ontology.

        Returns
        -------
            root_node: If _switch_label_and_id is True, the root node label is
                returned, otherwise the root node id is returned.

        """
        root_node = None
        root_label = self._root_label.replace("_", " ")

        if self._switch_label_and_id:
            root_node = to_lower_sentence_case(root_label)
        elif not self._switch_label_and_id:
            for node, data in self.get_nx_graph().nodes(data=True):
                if "label" in data and data["label"] == to_lower_sentence_case(root_label):
                    root_node = node
                    break

        return root_node

    def get_ancestors(self, node_label):
        """Get the ancestors of a node in the ontology."""
        return nx.dfs_preorder_nodes(self._nx_graph, node_label)

    def get_head_join_node(self):
        """Get the head join node of the ontology."""
        return self._head_join_node


class Ontology:
    """A class that represents the ontological "backbone" of a KG.

    The ontology can be built from a single resource, or hybridised from a
    combination of resources, with one resource being the "head" ontology, while
    an arbitrary number of other resources can become "tail" ontologies at
    arbitrary fusion points inside the "head" ontology.
    """

    def __init__(
        self,
        head_ontology: dict,
        ontology_mapping: Optional["OntologyMapping"] = None,
        tail_ontologies: dict | None = None,
    ):
        """Initialize the Ontology class.

        Args:
        ----
            head_ontology (OntologyAdapter): The head ontology.

            tail_ontologies (list): A list of OntologyAdapters that will be
                added to the head ontology. Defaults to None.

        """
        self._head_ontology_meta = head_ontology
        self.mapping = ontology_mapping
        self._tail_ontology_meta = tail_ontologies

        self._tail_ontologies = None
        self._nx_graph = None

        # keep track of nodes that have been extended
        self._extended_nodes = set()

        self._main()

    def _main(self) -> None:
        """Instantiate the ontology.

        Loads the ontologies, joins them, and returns the hybrid ontology.
        Loads only the head ontology if nothing else is given. Adds user
        extensions and properties from the mapping.
        """
        self._load_ontologies()

        if self._tail_ontologies:
            for adapter in self._tail_ontologies.values():
                head_join_node = self._get_head_join_node(adapter)
                self._join_ontologies(adapter, head_join_node)
        else:
            self._nx_graph = self._head_ontology.get_nx_graph()

        if self.mapping:
            self._extend_ontology()

            # experimental: add connections of disjoint classes to entity
            # self._connect_biolink_classes()

            self._add_properties()

    def _load_ontologies(self) -> None:
        """For each ontology, load the OntologyAdapter object.

        Store it as an instance variable (head) or in an instance dictionary
        (tail).
        """
        logger.info("Loading ontologies...")

        self._head_ontology = OntologyAdapter(
            ontology_file=self._head_ontology_meta["url"],
            root_label=self._head_ontology_meta["root_node"],
            ontology_file_format=self._head_ontology_meta.get("format", None),
            switch_label_and_id=self._head_ontology_meta.get("switch_label_and_id", True),
        )

        if self._tail_ontology_meta:
            self._tail_ontologies = {}
            for key, value in self._tail_ontology_meta.items():
                self._tail_ontologies[key] = OntologyAdapter(
                    ontology_file=value["url"],
                    root_label=value["tail_join_node"],
                    head_join_node_label=value["head_join_node"],
                    ontology_file_format=value.get("format", None),
                    merge_nodes=value.get("merge_nodes", True),
                    switch_label_and_id=value.get("switch_label_and_id", True),
                )

    def _get_head_join_node(self, adapter: OntologyAdapter) -> str:
        """Try to find the head join node of the given ontology adapter.

        Find the node in the head ontology that is the head join node. If the
        join node is not found, the method will raise an error.

        Args:
        ----
            adapter (OntologyAdapter): The ontology adapter of which to find the
                join node in the head ontology.

        Returns:
        -------
            str: The head join node in the head ontology.

        Raises:
        ------
            ValueError: If the head join node is not found in the head ontology.

        """
        head_join_node = None
        user_defined_head_join_node_label = adapter.get_head_join_node()
        head_join_node_label_in_bc_format = to_lower_sentence_case(user_defined_head_join_node_label.replace("_", " "))

        if self._head_ontology._switch_label_and_id:
            head_join_node = head_join_node_label_in_bc_format
        elif not self._head_ontology._switch_label_and_id:
            for node_id, data in self._head_ontology.get_nx_graph().nodes(data=True):
                if "label" in data and data["label"] == head_join_node_label_in_bc_format:
                    head_join_node = node_id
                    break

        if head_join_node not in self._head_ontology.get_nx_graph().nodes:
            head_ontology = self._head_ontology._rdf_to_nx(
                self._head_ontology.get_rdf_graph(),
                self._head_ontology._root_label,
                self._head_ontology._switch_label_and_id,
                rename_nodes=False,
            )
            msg = (
                f"Head join node '{head_join_node}' not found in head ontology. "
                f"The head ontology contains the following nodes: {head_ontology.nodes}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return head_join_node

    def _join_ontologies(self, adapter: OntologyAdapter, head_join_node) -> None:
        """Join the present ontologies.

        Join two ontologies by adding the tail ontology as a subgraph to the
        head ontology at the specified join nodes.

        Args:
        ----
            adapter (OntologyAdapter): The ontology adapter of the tail ontology
                to be added to the head ontology.

        """
        if not self._nx_graph:
            self._nx_graph = self._head_ontology.get_nx_graph().copy()

        tail_join_node = adapter.get_root_node()
        tail_ontology = adapter.get_nx_graph()

        # subtree of tail ontology at join node
        tail_ontology_subtree = nx.dfs_tree(tail_ontology.reverse(), tail_join_node).reverse()

        # transfer node attributes from tail ontology to subtree
        for node in tail_ontology_subtree.nodes:
            tail_ontology_subtree.nodes[node].update(tail_ontology.nodes[node])

        # if merge_nodes is False, create parent of tail join node from head
        # join node
        if not adapter._merge_nodes:
            # add head join node from head ontology to tail ontology subtree
            # as parent of tail join node
            tail_ontology_subtree.add_node(
                head_join_node,
                **self._head_ontology.get_nx_graph().nodes[head_join_node],
            )
            tail_ontology_subtree.add_edge(tail_join_node, head_join_node)

        # else rename tail join node to match head join node if necessary
        elif tail_join_node != head_join_node:
            tail_ontology_subtree = nx.relabel_nodes(tail_ontology_subtree, {tail_join_node: head_join_node})

        # combine head ontology and tail subtree
        self._nx_graph = nx.compose(self._nx_graph, tail_ontology_subtree)

    def _extend_ontology(self) -> None:
        """Add the user extensions to the ontology.

        Tries to find the parent in the ontology, adds it if necessary, and adds
        the child and a directed edge from child to parent. Can handle multiple
        parents.
        """
        if not self._nx_graph:
            self._nx_graph = self._head_ontology.get_nx_graph().copy()

        for key, value in self.mapping.extended_schema.items():
            # If this class is either a root or a synonym.
            if not value.get("is_a"):
                # If it is a synonym.
                if self._nx_graph.has_node(value.get("synonym_for")):
                    continue

                # If this class is in the schema, but not in the loaded vocabulary.
                if not self._nx_graph.has_node(key):
                    msg = (
                        f"Node {key} not found in ontology, but also has no inheritance definition. Please check your "
                        "schema for spelling errors, first letter not in lower case, use of underscores, a missing "
                        "`is_a` definition (SubClassOf a root node), or missing labels in class or super-classes."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                # It is a root and it is in the loaded vocabulary.
                continue

            # It is not a root.
            parents = to_list(value.get("is_a"))
            child = key

            while parents:
                parent = parents.pop(0)

                if parent not in self._nx_graph.nodes:
                    self._nx_graph.add_node(parent)
                    self._nx_graph.nodes[parent]["label"] = sentencecase_to_pascalcase(parent)

                    # mark parent as user extension
                    self._nx_graph.nodes[parent]["user_extension"] = True
                    self._extended_nodes.add(parent)

                if child not in self._nx_graph.nodes:
                    self._nx_graph.add_node(child)
                    self._nx_graph.nodes[child]["label"] = sentencecase_to_pascalcase(child)

                    # mark child as user extension
                    self._nx_graph.nodes[child]["user_extension"] = True
                    self._extended_nodes.add(child)

                self._nx_graph.add_edge(child, parent)

                child = parent

    def _connect_biolink_classes(self) -> None:
        """Experimental: Adds edges from disjoint classes to the entity node."""
        if not self._nx_graph:
            self._nx_graph = self._head_ontology.get_nx_graph().copy()

        if "entity" not in self._nx_graph.nodes:
            return

        # biolink classes that are disjoint from entity
        disjoint_classes = [
            "frequency qualifier mixin",
            "chemical entity to entity association mixin",
            "ontology class",
            "relationship quantifier",
            "physical essence or occurrent",
            "gene or gene product",
            "subject of investigation",
        ]

        for node in disjoint_classes:
            if not self._nx_graph.nodes.get(node):
                self._nx_graph.add_node(node)
                self._nx_graph.nodes[node]["label"] = sentencecase_to_pascalcase(node)

            self._nx_graph.add_edge(node, "entity")

    def _add_properties(self) -> None:
        """Add properties to the ontology.

        For each entity in the mapping, update the ontology with the properties
        specified in the mapping. Updates synonym information in the graph,
        setting the synonym as the primary node label.
        """
        for key, value in self.mapping.extended_schema.items():
            if key in self._nx_graph.nodes:
                self._nx_graph.nodes[key].update(value)

            if value.get("synonym_for"):
                # change node label to synonym
                if value["synonym_for"] not in self._nx_graph.nodes:
                    msg = f"Node {value['synonym_for']} not found in ontology."
                    logger.error(msg)
                    raise ValueError(msg)

                self._nx_graph = nx.relabel_nodes(self._nx_graph, {value["synonym_for"]: key})

    def get_ancestors(self, node_label: str) -> list:
        """Get the ancestors of a node in the ontology.

        Args:
        ----
            node_label (str): The label of the node in the ontology.

        Returns:
        -------
            list: A list of the ancestors of the node.

        """
        return nx.dfs_tree(self._nx_graph, node_label)

    def show_ontology_structure(self, to_disk: str = None, full: bool = False):
        """Show the ontology structure using treelib or write to GRAPHML file.

        Args:
        ----
            to_disk (str): If specified, the ontology structure will be saved
                to disk as a GRAPHML file at the location (directory) specified
                by the `to_disk` string, to be opened in your favourite graph
                visualisation tool.

            full (bool): If True, the full ontology structure will be shown,
                including all nodes and edges. If False, only the nodes and
                edges that are relevant to the extended schema will be shown.

        """
        if not full and not self.mapping.extended_schema:
            msg = (
                "You are attempting to visualise a subset of the loaded"
                "ontology, but have not provided a schema configuration. "
                "To display a partial ontology graph, please provide a schema "
                "configuration file; to visualise the full graph, please use "
                "the parameter `full = True`.",
            )
            logger.error(msg)
            raise ValueError(msg)

        if not self._nx_graph:
            msg = "Ontology not loaded."
            logger.error(msg)
            raise ValueError(msg)

        if not self._tail_ontologies:
            msg = f"Showing ontology structure based on {self._head_ontology._ontology_file}"

        else:
            msg = f"Showing ontology structure based on {len(self._tail_ontology_meta) + 1} ontologies: "

        logger.info(msg)

        if not full:
            # set of leaves and their intermediate parents up to the root
            filter_nodes = set(self.mapping.extended_schema.keys())

            for node in self.mapping.extended_schema.keys():
                filter_nodes.update(self.get_ancestors(node).nodes)

            # filter graph
            G = self._nx_graph.subgraph(filter_nodes)

        else:
            G = self._nx_graph

        if not to_disk:
            # create tree
            tree = create_tree_visualisation(G)

            # add synonym information
            for node in self.mapping.extended_schema:
                if not isinstance(self.mapping.extended_schema[node], dict):
                    continue
                if self.mapping.extended_schema[node].get("synonym_for"):
                    tree.nodes[node].tag = f"{node} = {self.mapping.extended_schema[node].get('synonym_for')}"

            logger.info(f"\n{tree}")

            return tree

        else:
            # convert lists/dicts to strings for vis only
            for node in G.nodes:
                # rename node and use former id as label
                label = G.nodes[node].get("label")

                if not label:
                    label = node

                G = nx.relabel_nodes(G, {node: label})
                G.nodes[label]["label"] = node

                for attrib in G.nodes[label]:
                    if type(G.nodes[label][attrib]) in [list, dict]:
                        G.nodes[label][attrib] = str(G.nodes[label][attrib])

            path = os.path.join(to_disk, "ontology_structure.graphml")

            logger.info(f"Writing ontology structure to {path}.")

            nx.write_graphml(G, path)

            return True

    def get_dict(self) -> dict:
        """Return a dictionary representation of the ontology.

        The dictionary is compatible with a BioCypher node for compatibility
        with the Neo4j driver.
        """
        d = {
            "node_id": self._get_current_id(),
            "node_label": "BioCypher",
            "properties": {
                "schema": "self.ontology_mapping.extended_schema",
            },
        }

        return d

    def _get_current_id(self):
        """Instantiate a version ID for the current session.

        For now does simple versioning using datetime.

        Can later implement incremental versioning, versioning from
        config file, or manual specification via argument.
        """
        now = datetime.now()
        return now.strftime("v%Y%m%d-%H%M%S")

    def get_rdf_graph(self):
        """Return the merged RDF graph.

        Return the merged graph of all loaded ontologies (head and tails).
        """
        graph = self._head_ontology.get_rdf_graph()
        if self._tail_ontologies:
            for key, onto in self._tail_ontologies.items():
                assert type(onto) == OntologyAdapter
                # RDFlib uses the + operator for merging.
                graph += onto.get_rdf_graph()
        return graph
#
#
### END contents of "[~]/biocypher/biocypher/_ontology.py" (with the aim of making the "from ._ontology import Ontology" command below superfluous).


### START contents of "[~]/biocypher/biocypher/_translate.py" (with the aim of making the "from biocypher._translate import Translator" command below superfluous)
#
#
"""BioCypher 'translation' module.

Responsible for translating between the raw input data and the
BioCypherNode and BioCypherEdge objects.
"""

from collections.abc import Generator, Iterable
from typing import Any

from more_itertools import peekable

logger.debug(f"Loading module {__name__}.")

__all__ = ["Translator"]


class Translator:
    """Class responsible for exacting the translation process.

    Translation is configured in the schema_config.yaml file. Creates a mapping
    dictionary from that file, and, given nodes and edges, translates them into
    BioCypherNodes and BioCypherEdges. During this process, can also filter the
    properties of the entities if the schema_config.yaml file specifies a property
    whitelist or blacklist.

    Provides utility functions for translating between input and output labels
    and cypher queries.
    """

    def __init__(self, ontology: "Ontology", strict_mode: bool = False):
        """Initialise the translator.

        Args:
        ----
            ontology (Ontology): An Ontology object providing schema and mapping details.
            strict_mode:
                strict_mode (bool, optional): If True, enforces that every node and edge carries
                the required 'source', 'licence', and 'version' properties. Raises ValueError
                if these are missing. Defaults to False.


        """
        self.ontology = ontology
        self.strict_mode = strict_mode

        # record nodes without biolink type configured in schema_config.yaml
        self.notype = {}

        # mapping functionality for translating terms and queries
        self.mappings = {}
        self.reverse_mappings = {}

        self._update_ontology_types()

    def translate_entities(self, entities):
        entities = peekable(entities)
        if isinstance(entities.peek(), BioCypherEdge | BioCypherNode | BioCypherRelAsNode):
            translated_entities = entities
        elif len(entities.peek()) < 4:
            translated_entities = self.translate_nodes(entities)
        else:
            translated_entities = self.translate_edges(entities)
        return translated_entities

    def translate_nodes(
        self,
        node_tuples: Iterable,
    ) -> Generator[BioCypherNode, None, None]:
        """Translate input node representation.

        Translate the node tuples to a representation that conforms to the
        schema of the given BioCypher graph. For now requires explicit
        statement of node type on pass.

        Args:
        ----
            node_tuples (list of tuples): collection of tuples
                representing individual nodes by their unique id and a type
                that is translated from the original database notation to
                the corresponding BioCypher notation.

        """
        self._log_begin_translate(node_tuples, "nodes")

        for _id, _type, _props in node_tuples:
            # check for strict mode requirements
            required_props = ["source", "licence", "version"]

            if self.strict_mode:
                # rename 'license' to 'licence' in _props
                if _props.get("license"):
                    _props["licence"] = _props.pop("license")

                for prop in required_props:
                    if prop not in _props:
                        msg = (
                            f"Property `{prop}` missing from node {_id}. "
                            "Strict mode is enabled, so this is not allowed.",
                        )
                        logger.error(msg)
                        raise ValueError(msg)

            # find the node in leaves that represents ontology node type
            _ontology_class = self._get_ontology_mapping(_type)

            if _ontology_class:
                # filter properties for those specified in schema_config if any
                _filtered_props = self._filter_props(_ontology_class, _props)

                # preferred id
                _preferred_id = self._get_preferred_id(_ontology_class)

                yield BioCypherNode(
                    node_id=_id,
                    node_label=_ontology_class,
                    preferred_id=_preferred_id,
                    properties=_filtered_props,
                )

            else:
                self._record_no_type(_type, _id)

        self._log_finish_translate("nodes")

    def _get_preferred_id(self, _bl_type: str) -> str:
        """Return the preferred id for the given Biolink type.

        If the preferred id is not specified in the schema_config.yaml file,
        return "id".
        """
        return (
            self.ontology.mapping.extended_schema[_bl_type]["preferred_id"]
            if "preferred_id" in self.ontology.mapping.extended_schema.get(_bl_type, {})
            else "id"
        )

    def _filter_props(self, bl_type: str, props: dict) -> dict:
        """Filter properties for those specified in schema_config if any.

        If the properties are not specified in the schema_config.yaml file,
        return the original properties.
        """
        filter_props = self.ontology.mapping.extended_schema[bl_type].get("properties", {})

        # strict mode: add required properties (only if there is a whitelist)
        if self.strict_mode and filter_props:
            filter_props.update(
                {"source": "str", "licence": "str", "version": "str"},
            )

        exclude_props = self.ontology.mapping.extended_schema[bl_type].get("exclude_properties", [])

        if isinstance(exclude_props, str):
            exclude_props = [exclude_props]

        if filter_props and exclude_props:
            filtered_props = {k: v for k, v in props.items() if (k in filter_props.keys() and k not in exclude_props)}

        elif filter_props:
            filtered_props = {k: v for k, v in props.items() if k in filter_props.keys()}

        elif exclude_props:
            filtered_props = {k: v for k, v in props.items() if k not in exclude_props}

        else:
            return props

        missing_props = [k for k in filter_props.keys() if k not in filtered_props.keys()]
        # add missing properties with default values
        for k in missing_props:
            filtered_props[k] = None

        return filtered_props

    def translate_edges(
        self,
        edge_tuples: Iterable,
    ) -> Generator[BioCypherEdge | BioCypherRelAsNode, None, None]:
        """Translate input edge representation.

        Translate the edge tuples to a representation that conforms to the
        schema of the given BioCypher graph. For now requires explicit
        statement of edge type on pass.

        Args:
        ----
            edge_tuples (list of tuples):

                collection of tuples representing source and target of
                an interaction via their unique ids as well as the type
                of interaction in the original database notation, which
                is translated to BioCypher notation using the `leaves`.
                Can optionally possess its own ID.

        """
        self._log_begin_translate(edge_tuples, "edges")

        # legacy: deal with 4-tuples (no edge id)
        # TODO remove for performance reasons once safe
        edge_tuples = peekable(edge_tuples)
        if len(edge_tuples.peek()) == 4:
            edge_tuples = [(None, src, tar, typ, props) for src, tar, typ, props in edge_tuples]

        for _id, _src, _tar, _type, _props in edge_tuples:
            # check for strict mode requirements
            if self.strict_mode:
                if "source" not in _props:
                    msg = (
                        f"Edge {_id if _id else (_src, _tar)} does not have a `source` property."
                        " This is required in strict mode.",
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                if "licence" not in _props:
                    msg = (
                        f"Edge {_id if _id else (_src, _tar)} does not have a `licence` property."
                        " This is required in strict mode.",
                    )
                    logger.error(msg)
                    raise ValueError(msg)

            # match the input label (_type) to
            # an ontology label from schema_config
            bl_type = self._get_ontology_mapping(_type)

            if bl_type:
                # filter properties for those specified in schema_config if any
                _filtered_props = self._filter_props(bl_type, _props)

                rep = self.ontology.mapping.extended_schema[bl_type]["represented_as"]

                if rep == "node":
                    if _id:
                        # if it brings its own ID, use it
                        node_id = _id

                    else:
                        # source target concat
                        node_id = str(_src) + "_" + str(_tar) + "_" + "_".join(str(v) for v in _filtered_props.values())

                    n = BioCypherNode(
                        node_id=node_id,
                        node_label=bl_type,
                        properties=_filtered_props,
                    )

                    # directionality check TODO generalise to account for
                    # different descriptions of directionality or find a
                    # more consistent solution for indicating directionality
                    if _filtered_props.get("directed") == True:  # noqa: E712 (seems to not work without '== True')
                        l1 = "IS_SOURCE_OF"
                        l2 = "IS_TARGET_OF"

                    elif _filtered_props.get(
                        "src_role",
                    ) and _filtered_props.get("tar_role"):
                        l1 = _filtered_props.get("src_role")
                        l2 = _filtered_props.get("tar_role")

                    else:
                        l1 = l2 = "IS_PART_OF"

                    e_s = BioCypherEdge(
                        source_id=_src,
                        target_id=node_id,
                        relationship_label=l1,
                        # additional here
                    )

                    e_t = BioCypherEdge(
                        source_id=_tar,
                        target_id=node_id,
                        relationship_label=l2,
                        # additional here
                    )

                    yield BioCypherRelAsNode(n, e_s, e_t)

                else:
                    edge_label = self.ontology.mapping.extended_schema[bl_type].get("label_as_edge")

                    if edge_label is None:
                        edge_label = bl_type

                    yield BioCypherEdge(
                        relationship_id=_id,
                        source_id=_src,
                        target_id=_tar,
                        relationship_label=edge_label,
                        properties=_filtered_props,
                    )

            else:
                self._record_no_type(_type, (_src, _tar))

        self._log_finish_translate("edges")

    def _record_no_type(self, _type: Any, what: Any) -> None:
        """Record the type of a non-represented node or edge.

        In case of an entity that is not represented in the schema_config,
        record the type and the entity.
        """
        logger.error(f"No ontology type defined for `{_type}`: {what}")

        if self.notype.get(_type, None):
            self.notype[_type] += 1

        else:
            self.notype[_type] = 1

    def get_missing_biolink_types(self) -> dict:
        """Return a dictionary of non-represented types.

        The dictionary contains the type as the key and the number of
        occurrences as the value.
        """
        return self.notype

    @staticmethod
    def _log_begin_translate(_input: Iterable, what: str):
        n = f"{len(_input)} " if hasattr(_input, "__len__") else ""

        logger.debug(f"Translating {n}{what} to BioCypher")

    @staticmethod
    def _log_finish_translate(what: str):
        logger.debug(f"Finished translating {what} to BioCypher.")

    def _update_ontology_types(self):
        """Create a dictionary to translate from input to ontology labels.

        If multiple input labels, creates mapping for each.
        """
        self._ontology_mapping = {}

        for key, value in self.ontology.mapping.extended_schema.items():
            labels = value.get("input_label") or value.get("label_in_input")

            if isinstance(labels, str):
                self._ontology_mapping[labels] = key

            elif isinstance(labels, list):
                for label in labels:
                    self._ontology_mapping[label] = key

            if value.get("label_as_edge"):
                self._add_translation_mappings(labels, value["label_as_edge"])

            else:
                self._add_translation_mappings(labels, key)

    def _get_ontology_mapping(self, label: str) -> str | None:
        """Find the ontology class for the given input type.

        For each given input type ("input_label" or "label_in_input"), find the
        corresponding ontology class in the leaves dictionary (from the
        `schema_config.yam`).

        Args:
        ----
            label:
                The input type to find (`input_label` or `label_in_input` in
                `schema_config.yaml`).

        """
        # FIXME does not seem like a necessary function.
        # commented out until behaviour of _update_bl_types is fixed
        return self._ontology_mapping.get(label, None)

    def translate_term(self, term):
        """Translate a single term."""
        return self.mappings.get(term, None)

    def reverse_translate_term(self, term):
        """Reverse translate a single term."""
        return self.reverse_mappings.get(term, None)

    def translate(self, querystr):
        """Translate a cypher query.

        Only translates labels as of now.
        """
        for key in self.mappings:

            querystr = querystr.replace(":" + key, ":" + self.mappings[key])            

        return querystr

    def reverse_translate(self, querystr):      
        """Reverse translate a cypher query.

        Only translates labels as of now.
        """
        for key in self.reverse_mappings:
            a = ":" + key + ")"
            b = ":" + key + "]"
            # TODO this conditional probably does not cover all cases
            if a in querystr or b in querystr:              

                if isinstance(self.reverse_mappings[key], list):

                    msg = (
                        "Reverse translation of multiple inputs not "
                        "implemented yet. Many-to-one mappings are "
                        "not reversible. "
                        f"({key} -> {self.reverse_mappings[key]})",
                    )
                    logger.error(msg)
                    raise NotImplementedError(msg)

                else:

                    querystr = querystr.replace(                        
                        a,
                        ":" + self.reverse_mappings[key] + ")",
                    ).replace(b, ":" + self.reverse_mappings[key] + "]")

        return querystr

    def _add_translation_mappings(self, original_name, biocypher_name):
        """Add translation mappings for a label and name.

        We use here the PascalCase version of the BioCypher name, since
        sentence case is not useful for Cypher queries.
        """
        if isinstance(original_name, list):
            for on in original_name:
                self.mappings[on] = self.name_sentence_to_pascal(
                    biocypher_name,
                )
        else:
            self.mappings[original_name] = self.name_sentence_to_pascal(
                biocypher_name,
            )

        if isinstance(biocypher_name, list):
            for bn in biocypher_name:
                self.reverse_mappings[
                    self.name_sentence_to_pascal(
                        bn,
                    )
                ] = original_name
        else:
            self.reverse_mappings[
                self.name_sentence_to_pascal(
                    biocypher_name,
                )
            ] = original_name

    @staticmethod
    def name_sentence_to_pascal(name: str) -> str:
        """Convert a name in sentence case to pascal case."""
        # split on dots if dot is present
        if "." in name:
            return ".".join(
                [_misc.sentencecase_to_pascalcase(n) for n in name.split(".")],
            )
        else:
            return _misc.sentencecase_to_pascalcase(name)
#
#
### END contents of "[~]/biocypher/biocypher/_translate.py" (with the aim of making the "from biocypher._translate import Translator" command below superfluous).


### START contents of "[~]/biocypher/biocypher/output/connect/_neo4j_driver.py" (as definition of "class _Neo4jDriver")
#
#
"""
BioCypher 'online' mode. Handles connection and manipulation of a running DBMS.
"""

import itertools

from collections.abc import Iterable

logger.debug(f"Loading module {__name__}.")
__all__ = ["_Neo4jDriver"]


class _Neo4jDriver:
    """
    Manages a BioCypher connection to a Neo4j database using the
    internal ``Neo4jDriver`` class.

    Args:

        database_name (str): The name of the database to connect to.

        wipe (bool): Whether to wipe the database before importing.

        uri (str): The URI of the database.

        user (str): The username to use for authentication.

        password (str): The password to use for authentication.

        multi_db (bool): Whether to use multi-database mode.

        fetch_size (int): The number of records to fetch at a time.

        increment_version (bool): Whether to increment the version number.

        translator (Translator): The translator to use for mapping.

        querystr (str) ["querystr: str | None = None,"?]

    """

    def __init__(
        self,
        database_name: str,
        uri: str,
        user: str,
        password: str,
        multi_db: bool,
        translator: Translator,
        querystr: str | None = None,
        wipe: bool = False,
        fetch_size: int = 1000,
        increment_version: bool = True,
        force_enterprise: bool = False,
        **kwargs,
    ):
        self.translator = translator

        self._driver = Neo4jDriver(
            db_name=database_name,
            db_uri=uri,
            db_user=user,
            db_passwd=password,
            fetch_size=fetch_size,
            wipe=wipe,
            multi_db=multi_db,
            raise_errors=True,
            force_enterprise=force_enterprise,
            querystr=querystr,
        )

        # check for biocypher config in connected graph

        if wipe:
            self.init_db()

        if increment_version:
            # set new current version node
            self._update_meta_graph()

    def _update_meta_graph(self):
        """Update the BioCypher meta graph with version information.

        Requires APOC to be installed. If APOC is not available, this
        operation is skipped with a warning.
        """
        if not self._driver.has_apoc:
            logger.warning(
                "APOC plugin is not available. Skipping meta graph update. "
                "Install APOC to enable version tracking: https://neo4j.com/labs/apoc/"
            )
            return

        logger.info("Updating Neo4j meta graph.")

        # find current version node
        db_version = self._driver.query(
            "MATCH (v:BioCypher) WHERE NOT (v)-[:PRECEDES]->() RETURN v",
        )
        # add version node
        self.add_biocypher_nodes(self.translator.ontology)

        # connect version node to previous
        if db_version[0]:
            previous = db_version[0][0]
            previous_id = previous["v"]["id"]
            e_meta = BioCypherEdge(
                previous_id,
                self.translator.ontology.get_dict().get("node_id"),
                "PRECEDES",
            )
            self.add_biocypher_edges(e_meta)

    def init_db(self):
        """
        Used to initialise a property graph database by setting up new
        constraints. Wipe has been performed by the ``Neo4jDriver``
        class already.

        Todo:
            - set up constraint creation interactively depending on the
                need of the database
        """

        logger.info("Initialising database.")
        self._create_constraints()

    def _create_constraints(self):
        """
        Creates constraints on node types in the graph. Used for
        initial setup.

        Grabs leaves of the ``schema_config.yaml`` file and creates
        constraints on the id of all entities represented as nodes.
        """

        logger.info("Creating constraints for node types in config.")

        major_neo4j_version = int(self._get_neo4j_version().split(".")[0])
        # get structure
        for leaf in self.translator.ontology.mapping.extended_schema.items():
            label = _misc.sentencecase_to_pascalcase(leaf[0], sep=r"\s\.")
            if leaf[1]["represented_as"] == "node":
                if major_neo4j_version >= 5:
                    s = f"CREATE CONSTRAINT `{label}_id` " f"IF NOT EXISTS FOR (n:`{label}`) " "REQUIRE n.id IS UNIQUE"
                    self._driver.query(s)
                else:
                    s = f"CREATE CONSTRAINT `{label}_id` " f"IF NOT EXISTS ON (n:`{label}`) " "ASSERT n.id IS UNIQUE"
                    self._driver.query(s)

    def _get_neo4j_version(self):
        """Get neo4j version.

        Returns the Neo4j server version. If detection fails, defaults to
        "5.0.0" to use the newer syntax (which is more likely for new
        installations).
        """
        try:
            neo4j_version = self._driver.query(
                """
                    CALL dbms.components()
                    YIELD name, versions, edition
                    UNWIND versions AS version
                    RETURN version AS version
                """,
            )[0][0]["version"]
            return neo4j_version
        except Exception as e:
            logger.warning(
                f"Error detecting Neo4j version: {e}. "
                "Defaulting to version 5.0.0 syntax. "
                "If you're using Neo4j 4.x, this may cause errors."
            )
            return "5.0.0"

    def add_nodes(self, id_type_tuples: Iterable[tuple]) -> tuple:
        """
        Generic node adder method to add any kind of input to the graph via the
        :class:`biocypher.create.BioCypherNode` class. Employs translation
        functionality and calls the :meth:`add_biocypher_nodes()` method.

        Args:
            id_type_tuples (iterable of 3-tuple): for each node to add to
                the biocypher graph, a 3-tuple with the following layout:
                first, the (unique if constrained) ID of the node; second, the
                type of the node, capitalised or PascalCase and in noun form
                (Neo4j primary label, eg `:Protein`); and third, a dictionary
                of arbitrary properties the node should possess (can be empty).

        Returns:
            2-tuple: the query result of :meth:`add_biocypher_nodes()`
                - first entry: data
                - second entry: Neo4j summary.
        """

        bn = self.translator.translate_nodes(id_type_tuples)
        return self.add_biocypher_nodes(bn)

    def add_edges(self, id_src_tar_type_tuples: Iterable[tuple]) -> tuple:
        """
        Generic edge adder method to add any kind of input to the graph
        via the :class:`biocypher.create.BioCypherEdge` class. Employs
        translation functionality and calls the
        :meth:`add_biocypher_edges()` method.

        Args:

            id_src_tar_type_tuples (iterable of 5-tuple):

                for each edge to add to the biocypher graph, a 5-tuple
                with the following layout: first, the optional unique ID
                of the interaction. This can be `None` if there is no
                systematic identifier (which for many interactions is
                the case). Second and third, the (unique if constrained)
                IDs of the source and target nodes of the relationship;
                fourth, the type of the relationship; and fifth, a
                dictionary of arbitrary properties the edge should
                possess (can be empty).

        Returns:

            2-tuple: the query result of :meth:`add_biocypher_edges()`

                - first entry: data
                - second entry: Neo4j summary.
        """

        bn = self.translator.translate_edges(id_src_tar_type_tuples)
        return self.add_biocypher_edges(bn)

    def add_biocypher_nodes(
        self,
        nodes: Iterable[BioCypherNode],
        explain: bool = False,
        profile: bool = False,
    ) -> bool:
        """
        Accepts a node type handoff class
        (:class:`biocypher.create.BioCypherNode`) with id,
        label, and a dict of properties (passing on the type of
        property, ie, ``int``, ``str``, ...).

        The dict retrieved by the
        :meth:`biocypher.create.BioCypherNode.get_dict()` method is
        passed into Neo4j as a map of maps, explicitly encoding node id
        and label, and adding all other properties from the 'properties'
        key of the dict. The merge is performed via APOC, matching only
        on node id to prevent duplicates. The same properties are set on
        match and on create, irrespective of the actual event.

        Args:
            nodes:
                An iterable of :class:`biocypher.create.BioCypherNode` objects.
            explain:
                Call ``EXPLAIN`` on the CYPHER query.
            profile:
                Do profiling on the CYPHER query.

        Returns:
            True for success, False otherwise.

        Raises:
            RuntimeError: If APOC is not available and required for the operation.
        """

        try:
            nodes = _misc.to_list(nodes)

            entities = [node.get_dict() for node in nodes]

        except AttributeError:
            msg = "Nodes must have a `get_dict` method."
            logger.error(msg)

            raise ValueError(msg)

        # Check if APOC is available
        if not self._driver.has_apoc:
            msg = (
                "APOC plugin is required for adding nodes. "
                "Please install APOC in your Neo4j instance. "
                "See: https://neo4j.com/labs/apoc/"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"Merging {len(entities)} nodes.")

        entity_query = (
            "UNWIND $entities AS ent "
            "CALL apoc.merge.node([ent.node_label], "
            "{id: ent.node_id}, ent.properties, ent.properties) "
            "YIELD node "
            "RETURN node"
        )

        method = "explain" if explain else "profile" if profile else "query"

        result = getattr(self._driver, method)(
            entity_query,
            parameters={
                "entities": entities,
            },
        )

        logger.info("Finished merging nodes.")

        return result

    def add_biocypher_edges(
        self,
        edges: Iterable[BioCypherEdge],
        explain: bool = False,
        profile: bool = False,
    ) -> bool:
        """
        Accepts an edge type handoff class
        (:class:`biocypher.create.BioCypherEdge`) with source
        and target ids, label, and a dict of properties (passing on the
        type of property, ie, int, string ...).

        The individual edge is either passed as a singleton, in the case
        of representation as an edge in the graph, or as a 4-tuple, in
        the case of representation as a node (with two edges connecting
        to interaction partners).

        The dict retrieved by the
        :meth:`biocypher.create.BioCypherEdge.get_dict()` method is
        passed into Neo4j as a map of maps, explicitly encoding source
        and target ids and the relationship label, and adding all edge
        properties from the 'properties' key of the dict. The merge is
        performed via APOC, matching only on source and target id to
        prevent duplicates. The same properties are set on match and on
        create, irrespective of the actual event.

        Args:
            edges:
                An iterable of :class:`biocypher.create.BioCypherEdge` objects.
            explain:
                Call ``EXPLAIN`` on the CYPHER query.
            profile:
                Do profiling on the CYPHER query.

        Returns:
            `True` for success, `False` otherwise.
        """

        edges = _misc.ensure_iterable(edges)
        edges = itertools.chain(*(_misc.ensure_iterable(i) for i in edges))

        nodes = []
        rels = []

        try:
            for e in edges:
                if hasattr(e, "get_node"):
                    nodes.append(e.get_node())
                    rels.append(e.get_source_edge().get_dict())
                    rels.append(e.get_target_edge().get_dict())

                else:
                    rels.append(e.get_dict())

        except AttributeError:
            msg = "Edges and nodes must have a `get_dict` method."
            logger.error(msg)

            raise ValueError(msg)

        self.add_biocypher_nodes(nodes)
        logger.info(f"Merging {len(rels)} edges.")

        # Check if APOC is available
        if not self._driver.has_apoc:
            msg = (
                "APOC plugin is required for adding edges. "
                "Please install APOC in your Neo4j instance. "
                "See: https://neo4j.com/labs/apoc/"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # cypher query

        # merging only on the ids of the entities, passing the
        # properties on match and on create;
        # TODO add node labels?
        node_query = "UNWIND $rels AS r " "MERGE (src {id: r.source_id}) " "MERGE (tar {id: r.target_id}) "

        self._driver.query(node_query, parameters={"rels": rels})

        edge_query = (
            "UNWIND $rels AS r "
            "MATCH (src {id: r.source_id}) "
            "MATCH (tar {id: r.target_id}) "
            "WITH src, tar, r "
            "CALL apoc.merge.relationship"
            "(src, r.relationship_label, NULL, "
            "r.properties, tar, r.properties) "
            "YIELD rel "
            "RETURN rel"
        )

        method = "explain" if explain else "profile" if profile else "query"

        result = getattr(self._driver, method)(edge_query, parameters={"rels": rels})

        logger.info("Finished merging edges.")

        return result
#
#
### END contents of "[~]/biocypher/biocypher/output/connect/_neo4j_driver.py".


class DatabaseAgent:
    def __init__(
        self,
        connection_args: dict,
    ) -> None:
        """Create a DatabaseAgent analogous to the VectorDatabaseAgentMilvus class,
        which can return results from a database using a query engine. Currently 
        limited to Neo4j for development.

        Args:
        ----
            connection_args (dict): A dictionary of arguments to connect to the 
                database. Contains database name, URI, user, and password.

            conversation_factory (Callable): A function to create a conversation 
                for creating the KG query.

            use_reflexion (bool): Whether to use the ReflexionAgent to generate 
                the query.
        """
        self.connection_args = connection_args
        self.driver = None

    def connect(self) -> None:
        """Connect to the database and authenticate."""
        db_name = self.connection_args.get("db_name")
        uri = f"{self.connection_args.get('host')}:{self.connection_args.get('port')}"
        uri = "bolt://localhost:7687"
        #
        user = self.connection_args.get("user")
        password = self.connection_args.get("password")
        #
        self.driver = _Neo4jDriver(        
            user="neo4j",
            #
            password="...",
            #
            database_name=db_name or "neo4j",
            uri=uri,
            multi_db=False,
            translator=Translator,
        )

    def is_connected(self) -> bool:
        return self.driver is not None

    def _build_response(
        self,
        results: list[dict],
        cypher_query: str,
        results_num: int | None = 3,
    ) -> list[Document]:
        if len(results) == 0:
            return [
                Document(
                    page_content=(
                        "I didn't find any result in knowledge graph, "
                        f"but here is the query I used: {cypher_query}. "
                        "You can ask user to refine the question. "
                        "Note: please ensure to include the query in a code "
                        "block in your response so that the user can refine "
                        "their question effectively."
                    ),
                    metadata={"cypher_query": cypher_query},
                ),
            ]

        clipped_results = results[:results_num] if results_num > 0 else results
        results_dump = json.dumps(clipped_results)

        return [
            Document(
                page_content=(
                    "The results retrieved from knowledge graph are: "
                    f"{results_dump}. "
                    f"The query used is: {cypher_query}. "
                    "Note: please ensure to include the query in a code block "
                    "in your response so that the user can refine "
                    "their question effectively."
                ),
                metadata={"cypher_query": cypher_query},
            ),
        ]        


# END OF FILE.
