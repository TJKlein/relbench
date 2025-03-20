import duckdb
import pandas as pd
from relbench.base import Database, Table, EntityTask, TaskType
from relbench.metrics import (
    accuracy,
    multilabel_f1_micro,
    multilabel_f1_macro,
)


class SALTTask(EntityTask):

    task_type = TaskType.MULTILABEL_CLASSIFICATION
    time_col = "CREATIONTIMESTAMP"
    timedelta = pd.Timedelta(days=1)
    header_target_cols = [
        "SALESOFFICE",
        "SALESGROUP",
        "CUSTOMERPAYMENTTERMS",
        "SHIPPINGCONDITION",
        "HEADERINCOTERMSCLASSIFICATION",
    ]
    item_target_cols = ["PLANT", "SHIPPINGPOINT", "ITEMINCOTERMSCLASSIFICATION"]

    metrics = [accuracy, multilabel_f1_micro, multilabel_f1_macro]

    def __init__(self, target_col, dataset, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.target_col = target_col

        # self.num_eval_timestamps = (
        #     self.dataset.get_db().max_timestamp - self.dataset.test_timestamp + 1
        # )

        # Make sure to remove additional targets from the sales order and sales order item tables to avoid leakers

        self.dataset.get_db().table_dict["salesdocumentitem"].df = (
            self.dataset.get_db()
            .table_dict["salesdocumentitem"]
            .df.drop([c for c in self.item_target_cols if c != self.target_col], axis=1)
        )
        self.dataset.get_db().table_dict["salesdocument"].df = (
            self.dataset.get_db()
            .table_dict["salesdocument"]
            .df.drop(
                [c for c in self.header_target_cols if c != self.target_col], axis=1
            )
        )

        self.num_labels = (
            self.dataset.get_db().table_dict[self.entity_table].df[target_col].nunique()
        )

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        df = db.table_dict[self.entity_table].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        query = f"""
            SELECT 
                timestamp,
                {self.entity_col},
                {self.target_col}
            FROM 
                timestamp_df,
                df
            WHERE 
                df.{self.time_col} > timestamp AND 
                df.{self.time_col} <= timestamp + INTERVAL '{self.timedelta}'
        """

        result_df = duckdb.sql(query).df()

        return Table(
            df=result_df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class PLANTSaltTask(SALTTask):
    """
    Fill in missing values in sales order item fields
    """

    entity_col = "ID"
    entity_table = "salesdocumentitem"
    default_target_col = "PLANT"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.default_target_col, dataset=dataset, **kwargs)


class SHIPPINGPOINTSaltTask(SALTTask):
    """
    Fill in missing values in sales order item fields
    """

    entity_col = "ID"
    entity_table = "salesdocumentitem"
    default_target_col = "SHIPPINGPOINT"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.default_target_col, dataset=dataset, **kwargs)


class ITEMINCOTERMSCLASSIFICATIONSaltTask(SALTTask):
    """
    Fill in missing values in sales order item fields
    """

    entity_col = "ID"
    entity_table = "salesdocumentitem"
    default_target_col = "ITEMINCOTERMSCLASSIFICATION"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.default_target_col, dataset=dataset, **kwargs)


class SALESOFFICESaltTask(SALTTask):
    """
    Fill in missing values in sales order fields
    """

    entity_col = "SALESDOCUMENT"
    entity_table = "salesdocument"
    target_col = "SALESOFFICE"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.target_col, dataset=dataset, **kwargs)


class SALESGROUPSaltTask(SALTTask):
    """
    Fill in missing values in sales order fields
    """

    entity_col = "SALESDOCUMENT"
    entity_table = "salesdocument"
    target_col = "SALESGROUP"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.target_col, dataset=dataset, **kwargs)


class CUSTOMERPAYMENTTERMSSaltTask(SALTTask):
    """
    Fill in missing values in sales order fields
    """

    entity_col = "SALESDOCUMENT"
    entity_table = "salesdocument"
    target_col = "CUSTOMERPAYMENTTERMS"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.target_col, dataset=dataset, **kwargs)


class SHIPPINGCONDITIONSaltTask(SALTTask):
    """
    Fill in missing values in sales order fields
    """

    entity_col = "SALESDOCUMENT"
    entity_table = "salesdocument"
    target_col = "SHIPPINGCONDITION"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.target_col, dataset=dataset, **kwargs)


class HEADERINCOTERMSCLASSIFICATIONSaltTask(SALTTask):
    """
    Fill in missing values in sales order fields
    """

    entity_col = "SALESDOCUMENT"
    entity_table = "salesdocument"
    target_col = "HEADERINCOTERMSCLASSIFICATION"

    def __init__(self, dataset, **kwargs):
        super().__init__(target_col=self.target_col, dataset=dataset, **kwargs)
