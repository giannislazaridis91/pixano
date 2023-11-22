import os
import json
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
import time
from lancedb import LanceDBConnection
import duckdb
import pyarrow as pa
from pixano.utils import natural_key
from pixano.data import Dataset


class BaseSampler(ABC):
    def __init__(self, db: LanceDBConnection):
        self.db = db

    @abstractmethod
    def query(self) -> list:
        pass


class BaseTrainer(ABC):
    def __init__(self, db: LanceDBConnection, model, validation_data):
        self.db = db
        self.model = model
        self.validation_data = validation_data

    @abstractmethod
    def train(self):
        pass


class BaseAnnotator(ABC):
    def __init__(self, db: LanceDBConnection):
        self.db = db

    @abstractmethod
    def annotate(self):
        pass


class PixanoAnnotator(BaseAnnotator):
    def __init__(self, db: LanceDBConnection, type="timed"):
        self.type = type
        super().__init__(db)

    # this version ask user to press when done
    def pixano_annotator_user_input(self, round):
        # search active_learning table for items to annotate
        candidates = getTaggedIds(self.db, round)
        db_ddb = self.db.open_table("db").to_lance()
        not_annoted = duckdb.sql(
            f"select id from db_ddb where id in ({ddb_str(candidates)}) and label is NULL"
        ).limit(len(candidates))
        # loop while not everything is annoted
        while len(not_annoted) > 0:
            list_cand = [item["id"] for item in not_annoted.arrow().to_pylist()]
            print(f"{len(not_annoted)} items to annotate on round {round}:", list_cand)
            input_str = input(
                "Annotate and press 'Enter' when done, or any text to abort"
            )
            if input_str != "":
                break
            # need to reopen to read changes
            db_ddb = self.db.open_table("db").to_lance()
            not_annoted = duckdb.sql(
                f"select id from db_ddb where id in ({ddb_str(candidates)}) and label is NULL"
            ).limit(len(not_annoted))

    # this version checks database each second to continue
    def pixano_annotator_timer(self, round):
        # search active_learning table for items to annotate
        candidates = getTaggedIds(self.db, round)
        db_ddb = self.db.open_table("db").to_lance()
        not_annoted = duckdb.sql(
            f"select id from db_ddb where id in ({ddb_str(candidates)}) and label is NULL"
        ).limit(len(candidates))
        prev_nan = 0
        # loop while not everything is annoted
        while len(not_annoted) > 0:
            list_cand = [item["id"] for item in not_annoted.arrow().to_pylist()]
            if len(not_annoted) != prev_nan:
                print(
                    f"{len(not_annoted)} items to annotate on round {round}:", list_cand
                )
                prev_nan = len(not_annoted)
            time.sleep(1.0)
            # need to reopen to read changes
            db_ddb = self.db.open_table("db").to_lance()
            not_annoted = duckdb.sql(
                f"select id from db_ddb where id in ({ddb_str(candidates)}) and label is NULL"
            ).limit(len(not_annoted))

    def annotate(self, round):
        if self.type == "timed":
            self.pixano_annotator_timer(round)
        elif self.type == "input":
            self.pixano_annotator_user_input(round)
        else:
            self.pixano_annotator_timer(round)


class Learner:
    def __init__(
        self,
        db: LanceDBConnection,
        trainer: BaseTrainer,
        sampler: BaseSampler,
        custom_annotator: BaseAnnotator = None,
        new_al=False,
        verbose=0,
    ):
        self.db = db
        self.trainer = trainer
        self.sampler = sampler
        self.annotator = (
            custom_annotator
            if custom_annotator is not None
            else PixanoAnnotator(self.db)
        )
        self.verbose = verbose

        # get ids from db
        db_table = self.db.open_table("db")
        db_ddb = db_table.to_lance()
        all_ids = duckdb.sql("select id from db_ddb").to_arrow_table().to_pylist()
        ids = [it["id"] for it in all_ids]
        ids = sorted(ids, key=natural_key)

        # ensure "db" table has a "label" column
        if "label" not in db_table.schema.names:
            # Create label table
            label_table = db_ddb.to_table(columns=["id"])
            label_array = pa.array([""] * len(db_table), type=pa.string())
            label_table = label_table.append_column(
                pa.field("label", pa.string()), label_array
            )
            # Merge with db table
            db_ddb.merge(label_table, "id")
            # Update DatasetInfo
            dataset = Dataset(Path(self.db.uri))
            dataset.info.tables["main"][0]["fields"]["label"] = "str"
            dataset.save_info()

        if "active_learning" not in db.table_names() or new_al:
            self._createALtable(ids)

        # clean stats if new AL cycle
        if new_al:
            statfile = Path(self.db.uri) / "stats.json"
            statfile.unlink(missing_ok=True)

    def _createALtable(self, ids):
        # create Active learning table
        self.db.drop_table("active_learning", ignore_missing=True)
        active_learning_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("round", pa.int32())
                # pa.field("label", pa.string()),  # moved to db
            ]
        )
        active_learning_tbl = self.db.create_table(
            "active_learning", schema=active_learning_schema
        )

        # fill table
        al_data = [
            {
                "id": id,
                "round": -1
                # 'label': None,
            }
            for id in ids
        ]
        active_learning_tbl.add(al_data)

        # add active_learning in db.json > tables
        with open(Path(self.db.uri) / "db.json", "r") as dbjson_file:
            dbjson = json.load(dbjson_file)
        dbjson["tables"].setdefault("active_learning", [])
        dbjson["tables"]["active_learning"].append({
            "name": "active_learning",
            "fields": {"id": "str", "round": "int"},
            "source": type(self.sampler).__name__
        })
        with open(Path(self.db.uri) / "db.json", "w") as dbjson_file:
            json.dump(dbjson, dbjson_file)

    def _compute_al_stats(self):
        histogram = []
        counts = defaultdict(lambda: defaultdict(int))

        db_table = self.db.open_table("db").to_lance()
        al_table = self.db.open_table("active_learning").to_lance()
        #TODO ?bug à checker?: si je ne met pas round >= 0 on a plein de labels, ce qui est fort curieux...
        # peut être précedentes annotations ?
        join_table = duckdb.sql("select label, round from db_table left join al_table using (id)"
                                "where db_table.label is not NULL and al_table.round >= 0").to_arrow_table()
        for item in join_table.to_pylist():
            counts[item["round"]][item["label"]] += 1

        for round_id, round_stats in counts.items():
            for label_id, label_stats in round_stats.items():
                histogram.append(
                    {
                        "Label distribution": label_id,
                        "counts": label_stats,
                        "split": f"round {round_id}",
                    }
                )

        stats = {
                "name": "Label distribution",
                "type": "categorical",
                "histogram": histogram,
            }

        # seek existing
        try:
            with open(Path(self.db.uri) / "stats.json", "r", encoding="utf-8") as f:
                json_stats = json.load(f)
        except FileNotFoundError:
            json_stats = []
        found = False
        for i, stat in enumerate(json_stats):
            if stat['name'] == 'Label distribution':
                json_stats[i] = stats
                found = True
                break
        if not found:
            json_stats.append(stats)

        with open(Path(self.db.uri) / "stats.json", "w", encoding="utf-8") as f:
            json.dump(json_stats, f)

    def _update_score_stat(self, round, n_candidates, score):
        try:
            with open(Path(self.db.uri) / "stats.json", "r", encoding="utf-8") as f:
                json_stats = json.load(f)
        except FileNotFoundError:
            json_stats = []
        score_json = [stat for stat in json_stats if stat["name"] == "Scores"]
        if score_json == []:
            score_json = {"name": "Scores", "type": "categorical", "histogram": []}
            json_stats.append(score_json)
        else:
            score_json = score_json[0]

        # histogram require same attribute as name, "counts", and "split" ... it doesn't match with or needs
        # TODO: write another histogram type ? (histogram.svelte)
        """
        score_json['histogram'].append({
            "Scores": score,
            "counts": n_candidates,
            "split": "round "+str(round)
        })
        """
        score_json["histogram"].append(
            {"Scores": round, "counts": score, "split": str(n_candidates)}
        )
        with open(Path(self.db.uri) / "stats.json", "w", encoding="utf-8") as f:
            json.dump(json_stats, f)

    def tagRound(self, round, candidates):
        active_learning_tbl = self.db.open_table("active_learning")
        active_learning_tbl.update(
            where=f"id in ({ddb_str(candidates)})", values={"round": round}
        )
        print(f"Round {round} tagged")

    # in case annotation is aborted, we may want to untag the current round
    # we also unlabel it
    def untagRound(self, round):
        ids = getTaggedIds(self.db, round)
        active_learning_tbl = self.db.open_table("active_learning")
        active_learning_tbl.update(where=f"round = {round}", values={"round": -1})
        db_tbl = self.db.open_table("db")
        db_tbl.update(where=f"id in ({ddb_str(ids)})", values={"label": None})

    def query(self, round, n_candidates=10):
        t0 = time.time() if self.verbose else None
        candidates = self.sampler.query(n_candidates)
        print(f"query time: {time.time() - t0}") if self.verbose else None

        # tag them to annotate
        t0 = time.time() if self.verbose else None
        self.tagRound(round, candidates)
        print(f"tag time: {time.time() - t0}") if self.verbose else None

        return candidates

    def annotate(self, round):
        candidates = getTaggedIds(self.db, round)
        print(f"{len(candidates)} candidates on round {round}")

        # annotate
        t0 = time.time() if self.verbose else None
        self.annotator.annotate(round)
        print(f"annotate time (annotate): {time.time() - t0}") if self.verbose else None

        # compute stats
        t0 = time.time() if self.verbose else None
        self._compute_al_stats()
        print(f"annotate time (stats): {time.time() - t0}") if self.verbose else None

    def train(self, round, epochs=10, batch_size=100):
        t0 = time.time() if self.verbose else None
        result = self.trainer.train(epochs=epochs, batch_size=batch_size)
        print(f"training time: {time.time() - t0}") if self.verbose else None
        # create score stat plot
        n_cand = getNumCandidatesRound(self.db, round)
        self._update_score_stat(round, n_cand, result["score"])
        return result


# UTILITY FUNCTIONS


def ddb_str(input_list: list) -> str:
    """return a string usable in duckdb query, e.g. "... where X in ({ddb_str(input_list)})"

    Args:
        input_list (list): list of items (that can be stringified with str(item))

    Returns:
        str: str to use in duckdb where clause
    """
    return ", ".join(["'" + str(item) + "'" for item in input_list])


def getLastRound(db: LanceDBConnection) -> int:
    """seek database to return current last round

    Args:
        db (LanceDBConnection): database as LanceDB connection

    Returns:
        int: current last round
    """
    al = db.open_table("active_learning").to_lance()
    data = duckdb.sql("select distinct(round) from al where round is not NULL")
    rounds = data.df()
    rounds = rounds.loc[:,"round"].to_list()
    # rounds = [it["round"] for it in data["round"].to_arrow_table().to_pylist()]
    return max(rounds) if rounds != [] else -1


def getNumCandidatesRound(db: LanceDBConnection, round: int) -> int:
    """seek database to return number of items for round

    Args:
        db (LanceDBConnection): database as LanceDB connection

        round (int): round

    Returns:
        int: number of candidates
    """
    al = db.open_table("active_learning").to_lance()
    return (
        duckdb.sql(f"select count(id) as n from al where round = {round}")
        .to_arrow_table()
        .to_pylist()[0]["n"]
    )


def getTaggedIds(db: LanceDBConnection, round: int = None) -> list[str]:
    """return tagged ids for round, or current last round if round is None

    Args:
        db (LanceDBConnection): database as LanceDB connection
        round (int, optional): round. Defaults to None for current last round.

    Returns:
        list[str]: list of tagged ids
    """
    al = db.open_table("active_learning").to_lance()
    if round is None:
        round = getLastRound(db)
    data = duckdb.sql(f"select id from al where round = {round}")
    return [it["id"] for it in data.to_arrow_table().to_pylist()]


def getLabelledIds(db: LanceDBConnection, round: int = None) -> list[str]:
    """return labelled ids for round or in whole database

    Args:
        db (LanceDBConnection): database as LanceDB connection
        round (int, optional): round to seek. Defaults to None for whole database seek.

    Returns:
        list[str]: labelled ids
    """
    db_ddb = db.open_table("db").to_lance()
    if round is None:
        data = duckdb.sql("select id from db_ddb where label is not NULL")
    else:
        al = db.open_table("active_learning").to_lance()
        data = duckdb.sql(
            f"select id from al LEFT JOIN db_ddb USING (id) where al.round = {round} and db_ddb.label is not NULL"
        )
    return [it["id"] for it in data.to_arrow_table().to_pylist()]


def getUnlabelledIds(db: LanceDBConnection, split: str = None) -> list[str]:
    """return unlabelled ids in whole dataset, optionnaly filtered by split

    Args:
        db (LanceDBConnection): database as LanceDB connection
        split (str, optional): split to seek. Defaults to None.

    Returns:
        list[str]: unlabelled ids
    """
    db_ddb = db.open_table("db").to_lance()
    if split is None:
        data = duckdb.sql("select id from db_ddb where label is NULL")
    else:
        data = duckdb.sql(
            f"select id from db_ddb where label is NULL and split = '{split}'"
        )

    # import pdb
    # pdb.set_trace()
    return [it["id"] for it in data.to_arrow_table().to_pylist()]


def getLabels(db: LanceDBConnection, round: int = None) -> list[str]:
    """return labels for round or all labels

    Args:
        db (LanceDBConnection): database as LanceDB connection
        round (int, optional): round to seek. Defaults to None for whole database.

    Returns:
        list[str]: labels
    """
    db_ddb = db.open_table("db").to_lance()
    if round is None:
        data = duckdb.sql("select label from db_ddb where label is not NULL")
    else:
        al = db.open_table("active_learning").to_lance()
        data = duckdb.sql(
            f"select label from db_ddb LEFT JOIN al USING (id) where al.round = {round} and db_ddb.label is not NULL"
        )
    return [it["label"] for it in data.to_arrow_table().to_pylist()]


def custom_update(tbl, where: str, col, values: list):
    # adapted from original LanceTable.update
    # used to update several rows with different values in one pass
    # REQUIRED: len(list) == len(items returned by 'where' query)
    orig_data = tbl._dataset.to_table(filter=where).combine_chunks()
    if len(orig_data) == 0:
        return
    if len(values) != len(orig_data):
        raise ValueError(
            f"Length of ({where}) filter ({len(orig_data)}) doesn't match length of values ({len(values)})"
        )

    i = orig_data.column_names.index(col)
    if i < 0:
        raise ValueError(f"Column {col} does not exist")
    orig_data = orig_data.set_column(i, col, pa.array(values, type=orig_data[col].type))
    tbl.delete(where)
    tbl.add(orig_data, mode="append")
    # blank update (update round with same value) to fake _reset_dataset() and register_event("update")
    # tbl.update(where, values={'round': round})
    tbl._reset_dataset()
    # lancedb.utils.events.register_event("update")
    # --- we don't have access to this. events in lancedb are for diagnostic only so we will skip that for now
    # --- update func is still experimental, there should be better options later


def getDataset(db: LanceDBConnection):
    """return unlabelled ids in whole dataset, optionnaly filtered by split

    Args:
        db (LanceDBConnection): database as LanceDB connection
        split (str, optional): split to seek. Defaults to None.

    Returns:
        list[str]: unlabelled ids
    """
    db_ddb = db.open_table("db").to_lance()

    tr_X = [os.path.join( os.path.dirname(db_ddb.uri), "media", "image", "train", it["id"]) for it in duckdb.sql(
            f"select id from db_ddb where split = 'train'"
        ).to_arrow_table().to_pylist()]

    tr_Y = [it["label"] for it in duckdb.sql(
            f"select label from db_ddb where split = 'train'"
        ).to_arrow_table().to_pylist()]

    tr_lb_X = [os.path.join( os.path.dirname(db_ddb.uri), "media", "image", "train", it["id"]) for it in duckdb.sql(
            f"select id from db_ddb where label is not NULL and split = 'train'"
        ).to_arrow_table().to_pylist()]

    tr_lb_Y = [it["label"] for it in duckdb.sql(
            f"select label from db_ddb where split = 'train' and label is not NULL"
        ).to_arrow_table().to_pylist()]

    te_X = [os.path.join( os.path.dirname(db_ddb.uri), "media", "image", "train", it["id"]) for it in duckdb.sql(
            f"select id from db_ddb where label is not NULL and split = 'test'"
        ).to_arrow_table().to_pylist()]

    te_Y = [it["label"] for it in duckdb.sql(
            f"select label from db_ddb where split = 'test' and label is not NULL"
        ).to_arrow_table().to_pylist()]

    return tr_X,tr_Y,tr_lb_X,tr_lb_Y,te_X,te_Y