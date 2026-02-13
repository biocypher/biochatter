import random
import string
import pandas as pd
from enum import Enum, auto
from itertools import chain
from typing import Optional
from biocypher._logger import logger

logger.debug(f"Loading module {__name__}.")


class PoleAdapterNodeType(Enum):
    """
    Define types of nodes the adapter can provide.
    """

    PERSON = ":Person"
    OFFICER = ":Officer"
    LOCATION = ":Location"
    CRIME = ":Crime"
    PHONE_CALL = ":PhoneCall"
    OBJECT = ":Object"


class PoleAdapterPersonField(Enum):
    """
    Define possible fields the adapter can provide for persons.
    """

    ID = "_id"
    NAME = "name"
    SURNAME = "surname"
    NHS_NO = "nhs_no"
    PHONE = "phone"
    EMAIL = "email"


class PoleAdapterOfficerField(Enum):
    """
    Define possible fields the adapter can provide for officers.
    """

    ID = "_id"
    NAME = "name"
    SURNAME = "surname"
    RANK = "rank"
    BADGE_NUMBER = "badge_no"


class PoleAdapterLocationField(Enum):
    """
    Define possible fields the adapter can provide for locations.
    """

    ID = "_id"
    ADDRESS = "address"
    POSTCODE = "postcode"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"


class PoleAdapterCrimeField(Enum):
    """
    Define possible fields the adapter can provide for crimes.
    """

    ID = "_id"
    TYPE = "type"
    DATE = "date"
    LAST_OUTCOME = "last_outcome"
    HASH = "id"


class PoleAdapterPhoneCallField(Enum):
    """
    Define possible fields the adapter can provide for phone calls.
    """

    ID = "_id"
    TYPE = "call_type"
    DATE = "call_date"
    TIME = "call_time"
    DURATION = "call_duration"


class PoleAdapterObjectField(Enum):
    """
    Define possible fields the adapter can provide for objects.
    """

    ID = "_id"
    TYPE = "type"
    DESCRIPTION = "description"


class PoleAdapterEdgeType(Enum):
    """
    Define possible edges the adapter can provide.
    """

    LIVES_AT = "CURRENT_ADDRESS"
    KNOWS = "KNOWS"
    INVOLVED_IN = "INVOLVED_IN"
    MADE_CALL = "CALLED"  # abstract
    RECEIVED_CALL = "CALLER"  # abstract
    OCCURED_AT = "OCCURRED_AT"
    INVESTIGATED_BY = "INVESTIGATED_BY"
    PARTY_TO = "PARTY_TO"
    RELATED_TO = "FAMILY_REL"


class PoleAdapter:
    """
    Example BioCypher adapter. Generates nodes and edges for creating a
    knowledge graph.

    Args:
        node_types: List of node types to include in the result.
        node_fields: List of node fields to include in the result.
        edge_types: List of edge types to include in the result.
        edge_fields: List of edge fields to include in the result.
    """

    def __init__(
        self,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
    ):
        self._set_types_and_fields(node_types, node_fields, edge_types, edge_fields)
        self._data = self._read_csv()
        self._node_data = self._get_node_data()
        self._edge_data = self._get_edge_data()
        self._phone_data = self._get_phone_data()
        self._email_data = self._get_email_data()
        self._caller_data = self._get_caller_data()
        self._called_data = self._get_called_data()

        # # print unique _labels
        # print(f"Unique labels: {self._data['_labels'].unique()}")

        # # print unique _type
        # print(f"Unique types: {self._data['_type'].unique()}")

    def _read_csv(self):
        """
        Read data from CSV file.
        """
        logger.info("Reading data from CSV file.")

        # data = pd.read_csv("data/pole.csv", dtype=str)   #<-- Commented this line as at 2025-12-10 19:50GMT (HD), as at runtime I got the below FileNotFoundError:
        # ====
        # (biocypher_chatter) hammie@hammie-Default-string:~/Disciplines/AI/BioCypher_Chatter/biochatter$ python ./2025-12-10_BioChatterQuickstart.py 
        # INFO -- This is BioCypher v0.12.0.
        # INFO -- Logging into `biocypher-log/biocypher-20251210-193103.log`.
        # WARNING -- Running BioCypher without schema configuration.
        # INFO -- Reading data from CSV file.
        # Traceback (most recent call last):
        #   File "/home/hammie/Disciplines/AI/BioCypher_Chatter/biochatter/./2025-12-10_BioChatterQuickstart.py", line 67, in <module>
        #     adaper = PoleAdapter()
        #   File "/home/hammie/Disciplines/AI/BioCypher_Chatter/biochatter/pole/pole/adapters/pole_adapter.py", line 132, in __init__
        #     self._data = self._read_csv()
        #   File "/home/hammie/Disciplines/AI/BioCypher_Chatter/biochatter/pole/pole/adapters/pole_adapter.py", line 152, in _read_csv
        #     data = pd.read_csv("data/pole.csv", dtype=str)
        #   File "/home/hammie/miniforge3/envs/biocypher_chatter/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
        #     return _read(filepath_or_buffer, kwds)
        #   File "/home/hammie/miniforge3/envs/biocypher_chatter/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 620, in _read
        #     parser = TextFileReader(filepath_or_buffer, **kwds)
        #   File "/home/hammie/miniforge3/envs/biocypher_chatter/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
        #     self._engine = self._make_engine(f, self.engine)
        #   File "/home/hammie/miniforge3/envs/biocypher_chatter/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
        #     self.handles = get_handle(
        #   File "/home/hammie/miniforge3/envs/biocypher_chatter/lib/python3.10/site-packages/pandas/io/common.py", line 873, in get_handle
        #     handle = open(
        # FileNotFoundError: [Errno 2] No such file or directory: 'data/pole.csv'
        # (biocypher_chatter) hammie@hammie-Default-string:~/Disciplines/AI/BioCypher_Chatter/biochatter$ pwd
        # /home/hammie/Disciplines/AI/BioCypher_Chatter/biochatter
        # (biocypher_chatter) hammie@hammie-Default-string:~/Disciplines/AI/BioCypher_Chatter/biochatter$
        # ====
        #
        data = pd.read_csv("../pole/data/pole.csv", dtype=str)   #<-- This line 'new' code as at 2025-12-10 19:53GMT (HD)

        # screen the entire data frame for double quotes
        data = data.map(lambda x: x.replace('"', "") if isinstance(x, str) else x)

        # rename 'id' to 'hash'
        data.rename(columns={"id": "hash"}, inplace=True)

        return data

    def _get_node_data(self):
        """
        Get all rows that do not have a _type.
        """
        return self._data[self._data["_type"].isnull()]

    def _get_edge_data(self):
        """
        Get all rows that have a _type.
        """
        return self._data[self._data["_type"].notnull()]

    def _get_phone_data(self):
        """
        Subset to only phone ownership relationships.
        """
        return self._data[self._data["_type"] == "HAS_PHONE"][["_start", "_end"]]

    def _get_email_data(self):
        """
        Subset to only email ownership relationships.
        """
        return self._data[self._data["_type"] == "HAS_EMAIL"][["_start", "_end"]]

    def _get_caller_data(self):
        """
        Subset to only call initiator relationships.
        """
        return self._data[self._data["_type"] == "CALLER"][["_start", "_end"]]

    def _get_called_data(self):
        """
        Subset to only call receiver relationships.
        """
        return self._data[self._data["_type"] == "CALLED"][["_start", "_end"]]

    def _get_phone(self, _id):
        """
        Get phone number for person.
        """
        if not _id in self._phone_data["_start"].values:
            return None

        phone_id = self._phone_data[self._phone_data["_start"] == _id]["_end"].values[0]
        phone = self._data[self._data["_id"] == phone_id]["phoneNo"].values[0]
        return phone

    def _get_email(self, _id):
        """
        Get email address for person.
        """
        if not _id in self._email_data["_start"].values:
            return None

        email_id = self._email_data[self._email_data["_start"] == _id]["_end"].values[0]
        email = self._data[self._data["_id"] == email_id]["email_address"].values[0]
        return email

    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor.
        """

        logger.info("Generating nodes.")

        # nodes: tuples of id, type, fields
        for index, row in self._node_data.iterrows():
            if row["_labels"] not in self.node_types:
                continue

            _id = row["_id"]
            _type = row["_labels"]
            _props = row.to_dict()
            # could filter non-values here

            # special cases - processing
            if _type == PoleAdapterNodeType.PERSON.value:
                _props["phone"] = self._get_phone(_id)
                _props["email"] = self._get_email(_id)
            yield (
                _id,
                _type,
                _props,
            )

    def get_edges(self):
        """
        Returns a generator of edge tuples for edge types specified in the
        adapter constructor.

        Args:
            probability: Probability of generating an edge between two nodes.
        """

        logger.info("Generating edges.")

        # edges: tuples of rel_id, start_id, end_id, type, fields
        for index, row in self._edge_data.iterrows():
            if row["_type"] not in self.edge_types:
                continue

            _id = None
            _start = row["_start"]
            _end = row["_end"]
            _type = row["_type"]
            _props = {}
            # could filter non-values here

            # special cases - processing
            if _type == PoleAdapterEdgeType.MADE_CALL.value:
                # caller is phone, extend to person
                # start of caller is phone call, end is phone
                _call_id = _start
                _caller_id = self._get_phone_owner_id(_end)

                _end = _call_id
                _start = _caller_id
                _type = "MADE_CALL"

            elif _type == PoleAdapterEdgeType.RECEIVED_CALL.value:
                # called is phone, extend to person
                # start of called is phone call, end is phone
                _call_id = _start
                _called_id = self._get_phone_owner_id(_end)

                _end = _call_id
                _start = _called_id
                _type = "RECEIVED_CALL"

            yield (
                _id,
                _start,
                _end,
                _type,
                _props,
            )

    def _get_phone_owner_id(self, phone_id):
        """
        Get ID of caller from phone ID. Phone data has person (_start) and phone
        (_end).
        """
        if not phone_id in self._phone_data["_end"].values:
            return None

        person_id = self._phone_data[self._phone_data["_end"] == phone_id][
            "_start"
        ].values[0]
        return person_id

    def _set_types_and_fields(self, node_types, node_fields, edge_types, edge_fields):
        if node_types:
            self.node_types = [type.value for type in node_types]
        else:
            self.node_types = [type.value for type in PoleAdapterNodeType]

        if node_fields:
            self.node_fields = [field.value for field in node_fields]
        else:
            self.node_fields = [
                field.value
                for field in chain(
                    PoleAdapterPersonField,
                    PoleAdapterLocationField,
                )
            ]

        if edge_types:
            self.edge_types = [type.value for type in edge_types]
        else:
            self.edge_types = [type.value for type in PoleAdapterEdgeType]

        if edge_fields:
            self.edge_fields = [field.value for field in edge_fields]
        else:
            self.edge_fields = [field.value for field in chain()]
