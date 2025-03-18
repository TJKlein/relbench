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

    metrics = [accuracy, multilabel_f1_micro, multilabel_f1_macro]

    def __init__(self, target_col, dataset, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.target_col = target_col
        self.num_labels = (
            self.dataset.get_db().table_dict[self.entity_table].df[target_col].nunique()
        )

    def make_table(
        self, db: Database, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp
    ) -> Table:
        df = db.table_dict[self.entity_table].df
        df = df[
            (df[self.time_col] >= start_timestamp) & (df[self.time_col] < end_timestamp)
        ][[self.entity_col, self.time_col, self.target_col]]

        return Table(
            df=df,
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
