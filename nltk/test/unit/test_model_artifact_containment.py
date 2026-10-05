"""GHSA-8mgp-746c-j5xp backport: real model I/O stays inside data roots."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nltk.data
from nltk import pathsec
from nltk.chunk.named_entity import Maxent_NE_Chunker
from nltk.classify.maxent import load_maxent_params, save_maxent_params
from nltk.parse import DependencyGraph
from nltk.parse.transitionparser import TransitionParser
from nltk.tag.perceptron import AveragedPerceptron, PerceptronTagger

WEIGHTS = {"feature": {"NN": 1.0}}


@pytest.fixture()
def sandbox(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    # The private system temp directory is itself allowed on some platforms.
    with tempfile.TemporaryDirectory(
        prefix=".nltk-model-backport-", dir=Path.home()
    ) as base:
        root = Path(base) / "data"
        root.mkdir(mode=0o700)
        outside = Path(base) / "outside"
        outside.mkdir(mode=0o700)
        monkeypatch.setattr(nltk.data, "path", [str(root)])
        monkeypatch.setenv("NLTK_DATA", "")
        monkeypatch.setattr(pathsec, "ENFORCE", True)
        monkeypatch.setattr(pathsec, "_ALLOWED_ROOTS_CACHE", None)
        monkeypatch.setattr(pathsec, "_LAST_DATA_PATHS", None)
        with pytest.raises(PermissionError):
            pathsec.validate_path(str(outside))
        yield root, outside


def _tagger() -> PerceptronTagger:
    tagger = PerceptronTagger(load=False)
    tagger.model.weights = WEIGHTS
    tagger.classes = {"NN"}
    return tagger


def _params() -> (
    tuple[np.ndarray, dict[tuple[str, str, str], int], list[str], dict[str, int]]
):
    return np.array([1.0]), {("feature", "value", "NN"): 0}, ["NN"], {"NN": 0}


@pytest.mark.parametrize("method", ["save", "load"])
def test_perceptron_refuses_outside_root(
    sandbox: tuple[Path, Path], method: str
) -> None:
    _, outside = sandbox
    target = outside / "weights.json"
    original = json.dumps(WEIGHTS)
    target.write_text(original, encoding="utf-8")
    with pytest.raises(PermissionError):
        pathsec.open(target)
    with pytest.raises(PermissionError):
        getattr(AveragedPerceptron(WEIGHTS), method)(target)
    assert target.read_text(encoding="utf-8") == original


def test_perceptron_authorized_round_trip(sandbox: tuple[Path, Path]) -> None:
    root, _ = sandbox
    target = root / "weights.json"
    AveragedPerceptron(WEIGHTS).save(target)
    loaded = AveragedPerceptron()
    loaded.load(target)
    assert loaded.weights == WEIGHTS


@pytest.mark.parametrize(
    "name", ["http://../../escape", " HTTPS://../../escape", "file:///etc/passwd"]
)
def test_model_path_is_not_a_url(sandbox: tuple[Path, Path], name: str) -> None:
    with pytest.raises(ValueError, match="Security Violation"):
        AveragedPerceptron(WEIGHTS).save(name)


def test_symlink_write_refused(sandbox: tuple[Path, Path]) -> None:
    root, outside = sandbox
    victim = outside / "victim.json"
    victim.write_text("untouched", encoding="utf-8")
    link = root / "link.json"
    try:
        link.symlink_to(victim)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PermissionError):
        AveragedPerceptron(WEIGHTS).save(link)
    assert victim.read_text(encoding="utf-8") == "untouched"


def test_tagger_save_cannot_create_or_authorize_outside_dir(
    sandbox: tuple[Path, Path],
) -> None:
    _, outside = sandbox
    target = outside / "new-model"
    before = list(nltk.data.path)
    with pytest.raises(PermissionError):
        _tagger().save_to_json(lang="eng", loc=target)
    assert not target.exists()
    assert nltk.data.path == before


def test_tagger_default_save_and_load(sandbox: tuple[Path, Path]) -> None:
    root, _ = sandbox
    tagger = _tagger()
    tagger.save_to_json(lang="eng")
    target = Path(tagger.save_dir)
    assert target.is_relative_to(root)
    if os.name == "posix":
        assert target.stat().st_mode & 0o077 == 0
    loaded = PerceptronTagger(lang="eng", loc=target)
    assert loaded.model.weights == WEIGHTS
    assert loaded.classes == {"NN"}


def test_tagger_load_does_not_widen_sandbox(sandbox: tuple[Path, Path]) -> None:
    _, outside = sandbox
    tagger = _tagger()
    tagger.save_to_json(lang="eng")
    planted = outside / "planted"
    shutil.copytree(tagger.save_dir, planted)
    before = list(nltk.data.path)
    with pytest.raises(PermissionError):
        PerceptronTagger(lang="eng", loc=planted)
    assert nltk.data.path == before


@pytest.mark.parametrize(
    "lang", ["../escape", "a/b", "a\\b", "C:escape", "\x00", ".", ".."]
)
def test_tagger_language_cannot_steer_output(
    sandbox: tuple[Path, Path], lang: str
) -> None:
    root, _ = sandbox
    target = root / "model"
    with pytest.raises(ValueError, match="Unsafe language"):
        _tagger().save_to_json(lang=lang, loc=target)
    assert not target.exists()


def test_maxent_outside_root_refused_before_creation(
    sandbox: tuple[Path, Path],
) -> None:
    _, outside = sandbox
    target = outside / "new-model"
    with pytest.raises(PermissionError):
        save_maxent_params(*_params(), tab_dir=target)
    assert not target.exists()


@pytest.mark.parametrize("use_default", [False, True])
def test_maxent_authorized_round_trip(
    sandbox: tuple[Path, Path], use_default: bool
) -> None:
    root, _ = sandbox
    params = _params()
    target = save_maxent_params(
        *params, tab_dir=None if use_default else root / "model"
    )
    assert Path(target).is_relative_to(root)
    loaded = load_maxent_params(nltk.data.FileSystemPathPointer(target))
    np.testing.assert_array_equal(loaded[0], params[0])
    assert loaded[1:] == params[1:]


def test_named_entity_save_uses_authorized_default(sandbox: tuple[Path, Path]) -> None:
    root, _ = sandbox
    weights, mapping, labels, alwayson = _params()
    encoding = SimpleNamespace(_mapping=mapping, _labels=labels, _alwayson=alwayson)
    chunker = object.__new__(Maxent_NE_Chunker)
    chunker._tagger = SimpleNamespace(
        _classifier=SimpleNamespace(_encoding=encoding, _weights=weights)
    )
    target = chunker.save_params()
    assert Path(target).is_relative_to(root)
    assert (Path(target) / "weights.txt").is_file()


def test_transition_parser_real_training_containment(
    sandbox: tuple[Path, Path],
) -> None:
    root, outside = sandbox
    graph = DependencyGraph(
        "Economic\tJJ\t2\tATT\nnews\tNN\t3\tSBJ\nhas\tVBD\t0\tROOT\n"
        "little\tJJ\t5\tATT\neffect\tNN\t3\tOBJ\non\tIN\t5\tATT\n"
        "financial\tJJ\t8\tATT\nmarkets\tNNS\t6\tPC\n.\t.\t3\tPU\n"
    )
    parser = TransitionParser("arc-standard")
    allowed = root / "model.pickle"
    parser.train([graph], str(allowed), verbose=False)
    assert parser.parse([], str(allowed)) == []
    planted = outside / "model.pickle"
    shutil.copyfile(allowed, planted)
    with pytest.raises(PermissionError):
        parser.parse([], str(planted))
    target = outside / "new.pickle"
    with pytest.raises(PermissionError):
        TransitionParser("arc-standard").train([graph], str(target), verbose=False)
    assert not target.exists()
