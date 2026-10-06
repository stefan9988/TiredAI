import dataclasses
import json
import subprocess
from types import SimpleNamespace

import pytest
from conftest import FakeEncoder, ToolCallingModel, raw_frame, tool_call
from langchain_core.messages import AIMessage
from langfuse import Evaluation
from langfuse.api import NotFoundError
from qdrant_client import QdrantClient

from tiredai import tracing
from tiredai.agent import build_agent
from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.conversations import ConversationTask, conversation_evaluator
from tiredai.benchmarks.retrieval import RankingTask, RetrievalQuery, ranking_evaluator
from tiredai.config import Settings
from tiredai.documents import products
from tiredai.preprocessing import normalize
from tiredai.search import CatalogSearch, make_search_tool
from tiredai.vectorstore import index_products


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("openrouter:qwen/qwen3-embedding-8b", ("openrouter", "qwen/qwen3-embedding-8b")),
        ("openrouter:nvidia/nemotron-3-embed-1b:free", ("openrouter", "nvidia/nemotron-3-embed-1b:free")),
        ("fastembed:BAAI/bge-small-en-v1.5", ("fastembed", "BAAI/bge-small-en-v1.5")),
        ("nvidia/nemotron-3-embed-1b:free", ("default", "nvidia/nemotron-3-embed-1b:free")),  # 'nvidia' is no provider
    ],
)
def test_embedding_models_are_given_with_an_optional_provider(spec, expected):
    assert ex.parse_embedding(spec, "default") == expected


def test_models_are_switched_on_a_copy_of_the_settings():
    settings = Settings.load()

    switched = ex.with_llm(ex.with_embedding(settings, "fastembed:BAAI/bge-small-en-v1.5"), "qwen/qwen3-30b-a3b-instruct-2507")

    assert (switched.embedding_provider, switched.embedding_model) == ("fastembed", "BAAI/bge-small-en-v1.5")
    assert switched.llm.model == "qwen/qwen3-30b-a3b-instruct-2507"
    assert switched.llm.system_prompt_path == settings.llm.system_prompt_path
    assert ex.with_llm(settings, None) is settings and ex.with_embedding(settings, None) is settings


class FakeLangfuse:
    """The dataset calls sync_dataset makes, against items kept in memory."""

    def __init__(self, *items, exists=True):
        self.items = {i.id: i for i in items}
        self.exists = exists
        self.created, self.upserts = [], []
        self.api = SimpleNamespace(datasets=SimpleNamespace(get=self._get))

    def _get(self, dataset_name):
        if not self.exists:
            raise NotFoundError(body="not found")

    def create_dataset(self, name, description):
        self.exists = True
        self.created.append(name)

    def create_dataset_item(self, *, dataset_name, id, input, expected_output, metadata, status):
        self.upserts.append((id, status))
        self.items[id] = SimpleNamespace(id=id, input=input, expected_output=expected_output, metadata=metadata, status=status)

    def get_dataset(self, name):
        return SimpleNamespace(items=list(self.items.values()))


def case(case_id: str, query: str = "q") -> ex.Case:
    return ex.Case(id=case_id, input={"query": query}, expected_output={"relevant": {"sku": ["A"]}}, metadata={"case": case_id})


def stored(c: ex.Case, status="ACTIVE"):
    return SimpleNamespace(id=f"ds--{c.id}", input=c.input, expected_output=c.expected_output, metadata=c.metadata, status=status)


def test_a_missing_dataset_is_created_with_every_case():
    langfuse = FakeLangfuse(exists=False)

    items = ex.sync_dataset(langfuse, "ds", "test", [case("a"), case("b")])

    assert langfuse.created == ["ds"]
    assert langfuse.upserts == [("ds--a", "ACTIVE"), ("ds--b", "ACTIVE")]
    assert [i.id for i in items] == ["ds--a", "ds--b"]


def test_only_new_and_changed_cases_are_sent_and_removed_ones_are_archived():
    unchanged, changed, removed, archived = case("same"), case("changed"), case("removed"), case("back")
    langfuse = FakeLangfuse(stored(unchanged), stored(changed), stored(removed), stored(archived, "ARCHIVED"))

    items = ex.sync_dataset(langfuse, "ds", "test", [unchanged, case("changed", "new query"), case("new"), archived])

    assert langfuse.created == []
    assert langfuse.upserts == [("ds--changed", "ACTIVE"), ("ds--new", "ACTIVE"), ("ds--back", "ACTIVE"), ("ds--removed", "ARCHIVED")]
    assert [i.id for i in items] == ["ds--same", "ds--changed", "ds--new", "ds--back"]
    assert items[1].input == {"query": "new query"}


def item_result(case_id, output, **scores):
    return SimpleNamespace(item={"metadata": {"case": case_id}}, output=output,
                           evaluations=[Evaluation(name=n, value=v, comment=f"{n} note") for n, v in scores.items()])  # fmt: skip


def test_a_run_is_summarized_per_score_and_written_as_a_report(tmp_path):
    result = SimpleNamespace(
        run_name="run · now",
        dataset_run_url="https://langfuse.example/run",
        item_results=[item_result("a", {"x": 1}, hit=1.0, rr=0.5), item_result("b", {"x": 2}, hit=0.0)],
    )

    summary = ex.summarize("hybrid · model", result, items=3, details={"seconds_per_query": 0.25})
    md_path, json_path = ex.write_report("retrieval", "Retrieval benchmark", [summary], ["rr", "hit"], ["note"], directory=tmp_path)

    assert summary.scores == {"hit": 0.5, "rr": 0.5} and summary.failed == 1
    assert summary.results[0] == {"case": "a", "output": {"x": 1}, "scores": {"hit": 1.0, "rr": 0.5}, "comments": {"hit": "hit note", "rr": "rr note"}}
    table = md_path.read_text()
    assert "| run | rr | hit | failed | seconds_per_query |" in table
    assert "| hybrid · model | 0.500 | 0.500 | 1/3 | 0.250 |" in table
    assert "https://langfuse.example/run" in table
    assert json.loads(json_path.read_text())[0]["results"][1]["case"] == "b"


def test_without_langfuse_keys_an_experiment_runs_locally():
    catalog = Catalog([{"sku": "A", "name": "Accelera Phi-R 205/55R16 91V", "size": "205/55R16", "price": 59.93}])
    client = QdrantClient(":memory:")
    index_products(client, "benchmark", list(catalog), FakeEncoder())
    query = RetrievalQuery(id="phi", kind="product", query="accelera phi-r", relevant={"sku": ["A"]})

    assert not tracing.enabled()
    result = tracing.client().run_experiment(
        name="test", data=ex.local_items([query.case()]), task=RankingTask(CatalogSearch(client, "benchmark", FakeEncoder(), max_results=10)),
        evaluators=[ranking_evaluator(catalog)],
    )  # fmt: skip

    summary = ex.summarize("local", result, items=1)
    assert summary.failed == 0 and summary.scores["hit@1"] == 1.0
    client.close()


@pytest.fixture
def scores_sent(traces, monkeypatch):
    sent = []
    monkeypatch.setattr(traces.client, "create_score", lambda **score: sent.append(score))
    return sent


def test_each_case_is_a_trace_with_the_agents_turns_inside_and_its_scores(traces, scores_sent, tmp_path):
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    tool = make_search_tool(CatalogSearch(client, "tires", FakeEncoder(), max_results=20))
    model = ToolCallingModel(messages=iter([tool_call("search_tires", size="205/55R15"), AIMessage(content="The Accelera Phi-R is $59.93.")]))
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.\n")
    settings = Settings.load()
    settings = dataclasses.replace(settings, llm=dataclasses.replace(settings.llm, system_prompt_path=prompt))
    agent = build_agent(settings, model=model, tools=[tool])
    catalog = Catalog(products(normalize(raw_frame({}))))
    expect = {"intent": "size_search", "constraints": {"size": "205/55R15"}}
    data = [{"input": {"turns": ["205/55R15?"]}, "expected_output": {"turns": [expect]}, "metadata": {"case": "phi"}}]

    result = traces.client.run_experiment(name="Agent: test/model", data=data, task=ConversationTask(agent, {}),
                                          evaluators=[conversation_evaluator(catalog)])  # fmt: skip

    assert result.item_results[0].output["turns"][0]["answer"] == "The Accelera Phi-R is $59.93."
    [run] = traces.tree()
    assert run[0] == ("experiment-item-run", "span")
    task = next(child for child in run[1] if child[0][0] == "experiment-item-task")
    [turn] = task[1]
    assert turn[0] == ("answer-shopper-message", "span")
    [turn_span] = traces.named("answer-shopper-message")
    assert turn_span.attributes["langfuse.trace.tags"] == ("benchmark",)
    assert {s["name"] for s in scores_sent} == {"intent_accuracy", "retrieval_hit@3", "filters_applied", "constraint_correctness",
                                                "groundedness", "passed"}  # fmt: skip
    assert {s["trace_id"] for s in scores_sent} == {format(turn_span.context.trace_id, "032x")}
    client.close()


def test_runs_record_their_commit_and_whether_tracked_files_changed(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "first")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()

    (tmp_path / "result.json").write_text("[]")  # untracked: a result file, not a code change
    assert ex.git_revision(tmp_path) == commit
    (tmp_path / "code.py").write_text("x = 2\n")
    assert ex.git_revision(tmp_path) == f"{commit}-dirty"
    assert ex.git_revision(tmp_path / "missing") == "unknown"
