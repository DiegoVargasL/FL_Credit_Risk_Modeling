from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common_pipeline import (ID, BUREAU_ID, TARGET, setup_console, add_sample_arg, get_paths)

N_WORKERS = 4

WINDOWS_DAYS = [90, 180]
WINDOWS_MONTHS = [3, 6]

STATS = ["sum", "mean", "min", "max", "std"]

EXCLUDE_PREFIXES = ("CODE_GENDER",)

# Columnas de la tabla de application_train que son categóricas y no númericas
APP_CAT_COLS = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

# Columnas que debemos tratar como texto
TEXT_COLS = {"NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO",
             "CODE_REJECT_REASON", "CREDIT_ACTIVE", "CREDIT_TYPE", "STATUS"}

# Columnas que tendran las tablas de cada cliente para el entrenamiento de los modelos.
# Se uso como referencia de estas columnas la tabla de entrenamiento obtenida en el 
# Modelo de Riesgo Crediticio con el mismo dataset pero de manera centralizada.
FEATURE_COLUMNS_ALL = ['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'DAYS_EMPLOYED_ANOM',
 'DEBT_TO_INCOME', 'CREDIT_TO_GOODS', 'CREDIT_TO_INCOME', 'ANNUITY_TO_CREDIT', 'DAYS_EMPLOYED_PERC',
 'INCOME_PER_PERSON', 'EXT_SOURCES_MEAN', 'EXT_SOURCES_MIN', 'EXT_SOURCES_MAX', 'EXT_SOURCES_STD',
 'DOCUMENTS_PROVIDED', 'PREV_AMT_ANNUITY_SUM', 'PREV_AMT_ANNUITY_MEAN', 'PREV_AMT_ANNUITY_MAX',
 'PREV_AMT_ANNUITY_MIN', 'PREV_AMT_ANNUITY_STD', 'PREV_AMT_APPLICATION_SUM', 'PREV_AMT_APPLICATION_MEAN',
 'PREV_AMT_APPLICATION_MAX', 'PREV_AMT_APPLICATION_MIN', 'PREV_AMT_APPLICATION_STD', 'PREV_AMT_CREDIT_SUM',
 'PREV_AMT_CREDIT_MEAN', 'PREV_AMT_CREDIT_MAX', 'PREV_AMT_CREDIT_MIN', 'PREV_AMT_CREDIT_STD',
 'PREV_AMT_DOWN_PAYMENT_SUM', 'PREV_AMT_DOWN_PAYMENT_MEAN', 'PREV_AMT_DOWN_PAYMENT_MAX',
 'PREV_AMT_DOWN_PAYMENT_MIN', 'PREV_AMT_DOWN_PAYMENT_STD', 'PREV_AMT_GOODS_PRICE_SUM',
 'PREV_AMT_GOODS_PRICE_MEAN', 'PREV_AMT_GOODS_PRICE_MAX', 'PREV_AMT_GOODS_PRICE_MIN',
 'PREV_AMT_GOODS_PRICE_STD', 'PREV_CNT_PAYMENT_MEAN', 'PREV_CNT_PAYMENT_MAX', 'PREV_CNT_PAYMENT_SUM',
 'PREV_DAYS_DECISION_MEAN', 'PREV_DAYS_DECISION_MAX', 'PREV_DAYS_DECISION_MIN', 'PREV_RATE_DOWN_PAYMENT_MEAN',
 'PREV_RATE_DOWN_PAYMENT_MAX', 'PREV_DAYS_FIRST_DRAWING_MIN', 'PREV_DAYS_FIRST_DRAWING_MAX',
 'PREV_DAYS_FIRST_DUE_MIN', 'PREV_DAYS_FIRST_DUE_MAX', 'PREV_DAYS_LAST_DUE_1ST_VERSION_MIN',
 'PREV_DAYS_LAST_DUE_1ST_VERSION_MAX', 'PREV_DAYS_LAST_DUE_MIN', 'PREV_DAYS_LAST_DUE_MAX',
 'PREV_DAYS_TERMINATION_MIN', 'PREV_DAYS_TERMINATION_MAX', 'PREV_APP_CREDIT_RATIO_MEAN',
 'PREV_APP_CREDIT_RATIO_MAX', 'PREV_APP_CREDIT_RATIO_MIN', 'PREV_COUNT',
 'PREV_NAME_CONTRACT_STATUS_APPROVED_SHARE', 'PREV_NAME_CONTRACT_STATUS_CANCELED_SHARE',
 'PREV_NAME_CONTRACT_STATUS_REFUSED_SHARE', 'PREV_NAME_CONTRACT_STATUS_UNUSED_OFFER_SHARE',
 'PREV_NAME_CONTRACT_TYPE_CASH_LOANS_SHARE', 'PREV_NAME_CONTRACT_TYPE_CONSUMER_LOANS_SHARE',
 'PREV_NAME_CONTRACT_TYPE_REVOLVING_LOANS_SHARE', 'PREV_NAME_CONTRACT_TYPE_XNA_SHARE',
 'PREV_NAME_CLIENT_TYPE_NEW_SHARE', 'PREV_NAME_CLIENT_TYPE_REFRESHED_SHARE',
 'PREV_NAME_CLIENT_TYPE_REPEATER_SHARE', 'PREV_NAME_CLIENT_TYPE_XNA_SHARE', 'PREV_NAME_PORTFOLIO_CARDS_SHARE',
 'PREV_NAME_PORTFOLIO_CARS_SHARE', 'PREV_NAME_PORTFOLIO_CASH_SHARE', 'PREV_NAME_PORTFOLIO_POS_SHARE',
 'PREV_NAME_PORTFOLIO_XNA_SHARE', 'PREV_CODE_REJECT_REASON_CLIENT_SHARE', 'PREV_CODE_REJECT_REASON_HC_SHARE',
 'PREV_CODE_REJECT_REASON_LIMIT_SHARE', 'PREV_CODE_REJECT_REASON_SCO_SHARE',
 'PREV_CODE_REJECT_REASON_SCOFR_SHARE', 'PREV_CODE_REJECT_REASON_SYSTEM_SHARE',
 'PREV_CODE_REJECT_REASON_VERIF_SHARE', 'PREV_CODE_REJECT_REASON_XAP_SHARE',
 'PREV_CODE_REJECT_REASON_XNA_SHARE', 'PREV_AMT_CREDIT_MEAN_L90', 'PREV_APP_CREDIT_RATIO_MEAN_L90',
 'PREV_AMT_CREDIT_DIFF_L90', 'PREV_APP_CREDIT_RATIO_DIFF_L90', 'PREV_AMT_CREDIT_RATIO_L90',
 'PREV_APP_CREDIT_RATIO_RATIO_L90', 'PREV_AMT_CREDIT_MEAN_L180', 'PREV_APP_CREDIT_RATIO_MEAN_L180',
 'PREV_AMT_CREDIT_DIFF_L180', 'PREV_APP_CREDIT_RATIO_DIFF_L180', 'PREV_AMT_CREDIT_RATIO_L180',
 'PREV_APP_CREDIT_RATIO_RATIO_L180', 'PREV_N_APPLICATIONS_L90', 'PREV_N_APPLICATIONS_L180',
 'BUREAU_DAYS_CREDIT_MIN', 'BUREAU_DAYS_CREDIT_MAX', 'BUREAU_DAYS_CREDIT_MEAN',
 'BUREAU_CREDIT_DAY_OVERDUE_SUM', 'BUREAU_CREDIT_DAY_OVERDUE_MEAN', 'BUREAU_CREDIT_DAY_OVERDUE_MAX',
 'BUREAU_CREDIT_DAY_OVERDUE_MIN', 'BUREAU_CREDIT_DAY_OVERDUE_STD', 'BUREAU_DAYS_CREDIT_ENDDATE_MIN',
 'BUREAU_DAYS_CREDIT_ENDDATE_MAX', 'BUREAU_DAYS_CREDIT_ENDDATE_MEAN', 'BUREAU_AMT_CREDIT_SUM_SUM',
 'BUREAU_AMT_CREDIT_SUM_MEAN', 'BUREAU_AMT_CREDIT_SUM_MAX', 'BUREAU_AMT_CREDIT_SUM_MIN',
 'BUREAU_AMT_CREDIT_SUM_STD', 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN',
 'BUREAU_AMT_CREDIT_SUM_DEBT_MAX', 'BUREAU_AMT_CREDIT_SUM_DEBT_MIN', 'BUREAU_AMT_CREDIT_SUM_DEBT_STD',
 'BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM', 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN',
 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX', 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MIN',
 'BUREAU_AMT_CREDIT_SUM_OVERDUE_STD', 'BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX',
 'BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN', 'BUREAU_CNT_CREDIT_PROLONG_SUM', 'BUREAU_CNT_CREDIT_PROLONG_MAX',
 'BUREAU_DEBT_CREDIT_RATIO_MEAN', 'BUREAU_DEBT_CREDIT_RATIO_MAX', 'BUREAU_COUNT',
 'BUREAU_CREDIT_ACTIVE_ACTIVE_SHARE', 'BUREAU_CREDIT_ACTIVE_BAD_DEBT_SHARE',
 'BUREAU_CREDIT_ACTIVE_CLOSED_SHARE', 'BUREAU_CREDIT_ACTIVE_SOLD_SHARE',
 'BUREAU_CREDIT_TYPE_ANOTHER_TYPE_OF_LOAN_SHARE', 'BUREAU_CREDIT_TYPE_CAR_LOAN_SHARE',
 'BUREAU_CREDIT_TYPE_CASH_LOAN_NON_EARMARKED_SHARE', 'BUREAU_CREDIT_TYPE_CONSUMER_CREDIT_SHARE',
 'BUREAU_CREDIT_TYPE_CREDIT_CARD_SHARE', 'BUREAU_CREDIT_TYPE_INTERBANK_CREDIT_SHARE',
 'BUREAU_CREDIT_TYPE_LOAN_FOR_BUSINESS_DEVELOPMENT_SHARE',
 'BUREAU_CREDIT_TYPE_LOAN_FOR_PURCHASE_OF_SHARES_MARGIN_LENDING_SHARE',
 'BUREAU_CREDIT_TYPE_LOAN_FOR_THE_PURCHASE_OF_EQUIPMENT_SHARE',
 'BUREAU_CREDIT_TYPE_LOAN_FOR_WORKING_CAPITAL_REPLENISHMENT_SHARE', 'BUREAU_CREDIT_TYPE_MICROLOAN_SHARE',
 'BUREAU_CREDIT_TYPE_MOBILE_OPERATOR_LOAN_SHARE', 'BUREAU_CREDIT_TYPE_MORTGAGE_SHARE',
 'BUREAU_CREDIT_TYPE_REAL_ESTATE_LOAN_SHARE', 'BUREAU_CREDIT_TYPE_UNKNOWN_TYPE_OF_LOAN_SHARE',
 'BUREAU_CREDIT_DAY_OVERDUE_MEAN_L90', 'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN_L90',
 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN_L90', 'BUREAU_CREDIT_DAY_OVERDUE_DIFF_L90',
 'BUREAU_AMT_CREDIT_SUM_DEBT_DIFF_L90', 'BUREAU_AMT_CREDIT_SUM_OVERDUE_DIFF_L90',
 'BUREAU_CREDIT_DAY_OVERDUE_RATIO_L90', 'BUREAU_AMT_CREDIT_SUM_DEBT_RATIO_L90',
 'BUREAU_AMT_CREDIT_SUM_OVERDUE_RATIO_L90', 'BUREAU_CREDIT_DAY_OVERDUE_MEAN_L180',
 'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN_L180', 'BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN_L180',
 'BUREAU_CREDIT_DAY_OVERDUE_DIFF_L180', 'BUREAU_AMT_CREDIT_SUM_DEBT_DIFF_L180',
 'BUREAU_AMT_CREDIT_SUM_OVERDUE_DIFF_L180', 'BUREAU_CREDIT_DAY_OVERDUE_RATIO_L180',
 'BUREAU_AMT_CREDIT_SUM_DEBT_RATIO_L180', 'BUREAU_AMT_CREDIT_SUM_OVERDUE_RATIO_L180', 'BB_MONTHS_BALANCE_MIN',
 'BB_MONTHS_BALANCE_MAX', 'BB_SEVERITY_MEAN', 'BB_SEVERITY_MAX', 'BB_SEVERITY_STD', 'BB_SEVERITY_SUM',
 'BB_DELINQUENT_MEAN', 'BB_DELINQUENT_SUM', 'BB_COUNT', 'BB_STATUS_0_SHARE', 'BB_STATUS_1_SHARE',
 'BB_STATUS_2_SHARE', 'BB_STATUS_3_SHARE', 'BB_STATUS_4_SHARE', 'BB_STATUS_5_SHARE', 'BB_STATUS_C_SHARE',
 'BB_STATUS_X_SHARE', 'BB_LOAN_WORST_SEVERITY_MEAN', 'BB_LOAN_WORST_SEVERITY_MAX',
 'BB_LOAN_EVER_DELINQUENT_SHARE', 'BB_SEVERITY_MEAN_L3', 'BB_DELINQUENT_MEAN_L3', 'BB_SEVERITY_DIFF_L3',
 'BB_DELINQUENT_DIFF_L3', 'BB_SEVERITY_RATIO_L3', 'BB_DELINQUENT_RATIO_L3', 'BB_SEVERITY_MEAN_L6',
 'BB_DELINQUENT_MEAN_L6', 'BB_SEVERITY_DIFF_L6', 'BB_DELINQUENT_DIFF_L6', 'BB_SEVERITY_RATIO_L6',
 'BB_DELINQUENT_RATIO_L6', 'INST_DPD_SUM', 'INST_DPD_MEAN', 'INST_DPD_MAX', 'INST_DPD_MIN', 'INST_DPD_STD',
 'INST_DBD_SUM', 'INST_DBD_MEAN', 'INST_DBD_MAX', 'INST_DBD_MIN', 'INST_DBD_STD', 'INST_LATE_PAYMENT_MEAN',
 'INST_LATE_PAYMENT_SUM', 'INST_AMT_INSTALMENT_SUM', 'INST_AMT_INSTALMENT_MEAN', 'INST_AMT_INSTALMENT_MAX',
 'INST_AMT_PAYMENT_SUM', 'INST_AMT_PAYMENT_MEAN', 'INST_AMT_PAYMENT_MAX', 'INST_PAYMENT_RATIO_MEAN',
 'INST_PAYMENT_RATIO_MIN', 'INST_PAYMENT_RATIO_STD', 'INST_PAYMENT_DIFF_SUM', 'INST_PAYMENT_DIFF_MEAN',
 'INST_PAYMENT_DIFF_MAX', 'INST_PAYMENT_DIFF_MIN', 'INST_PAYMENT_DIFF_STD', 'INST_COUNT', 'INST_DPD_MEAN_L90',
 'INST_LATE_PAYMENT_MEAN_L90', 'INST_PAYMENT_RATIO_MEAN_L90', 'INST_PAYMENT_DIFF_MEAN_L90',
 'INST_DPD_DIFF_L90', 'INST_LATE_PAYMENT_DIFF_L90', 'INST_PAYMENT_RATIO_DIFF_L90',
 'INST_PAYMENT_DIFF_DIFF_L90', 'INST_DPD_RATIO_L90', 'INST_LATE_PAYMENT_RATIO_L90',
 'INST_PAYMENT_RATIO_RATIO_L90', 'INST_PAYMENT_DIFF_RATIO_L90', 'INST_DPD_MEAN_L180',
 'INST_LATE_PAYMENT_MEAN_L180', 'INST_PAYMENT_RATIO_MEAN_L180', 'INST_PAYMENT_DIFF_MEAN_L180',
 'INST_DPD_DIFF_L180', 'INST_LATE_PAYMENT_DIFF_L180', 'INST_PAYMENT_RATIO_DIFF_L180',
 'INST_PAYMENT_DIFF_DIFF_L180', 'INST_DPD_RATIO_L180', 'INST_LATE_PAYMENT_RATIO_L180',
 'INST_PAYMENT_RATIO_RATIO_L180', 'INST_PAYMENT_DIFF_RATIO_L180', 'POS_MONTHS_BALANCE_MIN',
 'POS_MONTHS_BALANCE_MAX', 'POS_SK_DPD_SUM', 'POS_SK_DPD_MEAN', 'POS_SK_DPD_MAX', 'POS_SK_DPD_MIN',
 'POS_SK_DPD_STD', 'POS_SK_DPD_DEF_SUM', 'POS_SK_DPD_DEF_MEAN', 'POS_SK_DPD_DEF_MAX', 'POS_SK_DPD_DEF_MIN',
 'POS_SK_DPD_DEF_STD', 'POS_DPD_FLAG_MEAN', 'POS_DPD_FLAG_SUM', 'POS_CNT_INSTALMENT_MEAN',
 'POS_CNT_INSTALMENT_MAX', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'POS_CNT_INSTALMENT_FUTURE_MIN', 'POS_COUNT',
 'POS_NAME_CONTRACT_STATUS_ACTIVE_SHARE', 'POS_NAME_CONTRACT_STATUS_AMORTIZED_DEBT_SHARE',
 'POS_NAME_CONTRACT_STATUS_APPROVED_SHARE', 'POS_NAME_CONTRACT_STATUS_CANCELED_SHARE',
 'POS_NAME_CONTRACT_STATUS_COMPLETED_SHARE', 'POS_NAME_CONTRACT_STATUS_DEMAND_SHARE',
 'POS_NAME_CONTRACT_STATUS_RETURNED_TO_THE_STORE_SHARE', 'POS_NAME_CONTRACT_STATUS_SIGNED_SHARE',
 'POS_NAME_CONTRACT_STATUS_XNA_SHARE', 'POS_SK_DPD_MEAN_L3', 'POS_SK_DPD_DEF_MEAN_L3', 'POS_DPD_FLAG_MEAN_L3',
 'POS_SK_DPD_DIFF_L3', 'POS_SK_DPD_DEF_DIFF_L3', 'POS_DPD_FLAG_DIFF_L3', 'POS_SK_DPD_RATIO_L3',
 'POS_SK_DPD_DEF_RATIO_L3', 'POS_DPD_FLAG_RATIO_L3', 'POS_SK_DPD_MEAN_L6', 'POS_SK_DPD_DEF_MEAN_L6',
 'POS_DPD_FLAG_MEAN_L6', 'POS_SK_DPD_DIFF_L6', 'POS_SK_DPD_DEF_DIFF_L6', 'POS_DPD_FLAG_DIFF_L6',
 'POS_SK_DPD_RATIO_L6', 'POS_SK_DPD_DEF_RATIO_L6', 'POS_DPD_FLAG_RATIO_L6', 'CC_MONTHS_BALANCE_MIN',
 'CC_MONTHS_BALANCE_MAX', 'CC_AMT_BALANCE_SUM', 'CC_AMT_BALANCE_MEAN', 'CC_AMT_BALANCE_MAX',
 'CC_AMT_BALANCE_MIN', 'CC_AMT_BALANCE_STD', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN',
 'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX', 'CC_AMT_DRAWINGS_CURRENT_SUM', 'CC_AMT_DRAWINGS_CURRENT_MEAN',
 'CC_AMT_DRAWINGS_CURRENT_MAX', 'CC_AMT_DRAWINGS_CURRENT_MIN', 'CC_AMT_DRAWINGS_CURRENT_STD',
 'CC_AMT_PAYMENT_CURRENT_SUM', 'CC_AMT_PAYMENT_CURRENT_MEAN', 'CC_AMT_PAYMENT_CURRENT_MAX',
 'CC_AMT_PAYMENT_CURRENT_MIN', 'CC_AMT_PAYMENT_CURRENT_STD', 'CC_AMT_TOTAL_RECEIVABLE_MEAN',
 'CC_AMT_TOTAL_RECEIVABLE_MAX', 'CC_SK_DPD_SUM', 'CC_SK_DPD_MEAN', 'CC_SK_DPD_MAX', 'CC_SK_DPD_MIN',
 'CC_SK_DPD_STD', 'CC_SK_DPD_DEF_SUM', 'CC_SK_DPD_DEF_MEAN', 'CC_SK_DPD_DEF_MAX', 'CC_SK_DPD_DEF_MIN',
 'CC_SK_DPD_DEF_STD', 'CC_UTILIZATION_MEAN', 'CC_UTILIZATION_MAX', 'CC_UTILIZATION_MIN', 'CC_UTILIZATION_STD',
 'CC_PAYMENT_MIN_RATIO_MEAN', 'CC_PAYMENT_MIN_RATIO_MIN', 'CC_COUNT', 'CC_NAME_CONTRACT_STATUS_ACTIVE_SHARE',
 'CC_NAME_CONTRACT_STATUS_APPROVED_SHARE', 'CC_NAME_CONTRACT_STATUS_COMPLETED_SHARE',
 'CC_NAME_CONTRACT_STATUS_DEMAND_SHARE', 'CC_NAME_CONTRACT_STATUS_REFUSED_SHARE',
 'CC_NAME_CONTRACT_STATUS_SENT_PROPOSAL_SHARE', 'CC_NAME_CONTRACT_STATUS_SIGNED_SHARE',
 'CC_UTILIZATION_MEAN_L3', 'CC_AMT_BALANCE_MEAN_L3', 'CC_SK_DPD_MEAN_L3', 'CC_PAYMENT_MIN_RATIO_MEAN_L3',
 'CC_UTILIZATION_DIFF_L3', 'CC_AMT_BALANCE_DIFF_L3', 'CC_SK_DPD_DIFF_L3', 'CC_PAYMENT_MIN_RATIO_DIFF_L3',
 'CC_UTILIZATION_RATIO_L3', 'CC_AMT_BALANCE_RATIO_L3', 'CC_SK_DPD_RATIO_L3', 'CC_PAYMENT_MIN_RATIO_RATIO_L3',
 'CC_UTILIZATION_MEAN_L6', 'CC_AMT_BALANCE_MEAN_L6', 'CC_SK_DPD_MEAN_L6', 'CC_PAYMENT_MIN_RATIO_MEAN_L6',
 'CC_UTILIZATION_DIFF_L6', 'CC_AMT_BALANCE_DIFF_L6', 'CC_SK_DPD_DIFF_L6', 'CC_PAYMENT_MIN_RATIO_DIFF_L6',
 'CC_UTILIZATION_RATIO_L6', 'CC_AMT_BALANCE_RATIO_L6', 'CC_SK_DPD_RATIO_L6', 'CC_PAYMENT_MIN_RATIO_RATIO_L6',
 'NAME_CONTRACT_TYPE_CASH_LOANS', 'NAME_CONTRACT_TYPE_REVOLVING_LOANS', 'NAME_CONTRACT_TYPE_NAN',
 'CODE_GENDER_F', 'CODE_GENDER_M', 'CODE_GENDER_NAN', 'FLAG_OWN_CAR_N', 'FLAG_OWN_CAR_Y', 'FLAG_OWN_CAR_NAN',
 'FLAG_OWN_REALTY_N', 'FLAG_OWN_REALTY_Y', 'FLAG_OWN_REALTY_NAN', 'NAME_TYPE_SUITE_CHILDREN',
 'NAME_TYPE_SUITE_FAMILY', 'NAME_TYPE_SUITE_GROUP_OF_PEOPLE', 'NAME_TYPE_SUITE_OTHER_A',
 'NAME_TYPE_SUITE_OTHER_B', 'NAME_TYPE_SUITE_SPOUSE_PARTNER', 'NAME_TYPE_SUITE_UNACCOMPANIED',
 'NAME_TYPE_SUITE_NAN', 'NAME_INCOME_TYPE_BUSINESSMAN', 'NAME_INCOME_TYPE_COMMERCIAL_ASSOCIATE',
 'NAME_INCOME_TYPE_MATERNITY_LEAVE', 'NAME_INCOME_TYPE_PENSIONER', 'NAME_INCOME_TYPE_STATE_SERVANT',
 'NAME_INCOME_TYPE_STUDENT', 'NAME_INCOME_TYPE_UNEMPLOYED', 'NAME_INCOME_TYPE_WORKING',
 'NAME_INCOME_TYPE_NAN', 'NAME_EDUCATION_TYPE_ACADEMIC_DEGREE', 'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
 'NAME_EDUCATION_TYPE_INCOMPLETE_HIGHER', 'NAME_EDUCATION_TYPE_LOWER_SECONDARY',
 'NAME_EDUCATION_TYPE_SECONDARY_SECONDARY_SPECIAL', 'NAME_EDUCATION_TYPE_NAN',
 'NAME_FAMILY_STATUS_CIVIL_MARRIAGE', 'NAME_FAMILY_STATUS_MARRIED', 'NAME_FAMILY_STATUS_SEPARATED',
 'NAME_FAMILY_STATUS_SINGLE_NOT_MARRIED', 'NAME_FAMILY_STATUS_UNKNOWN', 'NAME_FAMILY_STATUS_WIDOW',
 'NAME_FAMILY_STATUS_NAN', 'NAME_HOUSING_TYPE_CO_OP_APARTMENT', 'NAME_HOUSING_TYPE_HOUSE_APARTMENT',
 'NAME_HOUSING_TYPE_MUNICIPAL_APARTMENT', 'NAME_HOUSING_TYPE_OFFICE_APARTMENT',
 'NAME_HOUSING_TYPE_RENTED_APARTMENT', 'NAME_HOUSING_TYPE_WITH_PARENTS', 'NAME_HOUSING_TYPE_NAN',
 'OCCUPATION_TYPE_ACCOUNTANTS', 'OCCUPATION_TYPE_CLEANING_STAFF', 'OCCUPATION_TYPE_COOKING_STAFF',
 'OCCUPATION_TYPE_CORE_STAFF', 'OCCUPATION_TYPE_DRIVERS', 'OCCUPATION_TYPE_HR_STAFF',
 'OCCUPATION_TYPE_HIGH_SKILL_TECH_STAFF', 'OCCUPATION_TYPE_IT_STAFF', 'OCCUPATION_TYPE_LABORERS',
 'OCCUPATION_TYPE_LOW_SKILL_LABORERS', 'OCCUPATION_TYPE_MANAGERS', 'OCCUPATION_TYPE_MEDICINE_STAFF',
 'OCCUPATION_TYPE_PRIVATE_SERVICE_STAFF', 'OCCUPATION_TYPE_REALTY_AGENTS', 'OCCUPATION_TYPE_SALES_STAFF',
 'OCCUPATION_TYPE_SECRETARIES', 'OCCUPATION_TYPE_SECURITY_STAFF', 'OCCUPATION_TYPE_WAITERS_BARMEN_STAFF',
 'OCCUPATION_TYPE_NAN', 'WEEKDAY_APPR_PROCESS_START_FRIDAY', 'WEEKDAY_APPR_PROCESS_START_MONDAY',
 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY',
 'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY',
 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY', 'WEEKDAY_APPR_PROCESS_START_NAN', 'ORGANIZATION_TYPE_ADVERTISING',
 'ORGANIZATION_TYPE_AGRICULTURE', 'ORGANIZATION_TYPE_BANK', 'ORGANIZATION_TYPE_BUSINESS_ENTITY_TYPE_1',
 'ORGANIZATION_TYPE_BUSINESS_ENTITY_TYPE_2', 'ORGANIZATION_TYPE_BUSINESS_ENTITY_TYPE_3',
 'ORGANIZATION_TYPE_CLEANING', 'ORGANIZATION_TYPE_CONSTRUCTION', 'ORGANIZATION_TYPE_CULTURE',
 'ORGANIZATION_TYPE_ELECTRICITY', 'ORGANIZATION_TYPE_EMERGENCY', 'ORGANIZATION_TYPE_GOVERNMENT',
 'ORGANIZATION_TYPE_HOTEL', 'ORGANIZATION_TYPE_HOUSING', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_1',
 'ORGANIZATION_TYPE_INDUSTRY_TYPE_10', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_11',
 'ORGANIZATION_TYPE_INDUSTRY_TYPE_12', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_13',
 'ORGANIZATION_TYPE_INDUSTRY_TYPE_2', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_3',
 'ORGANIZATION_TYPE_INDUSTRY_TYPE_4', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_5',
 'ORGANIZATION_TYPE_INDUSTRY_TYPE_6', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_7',
 'ORGANIZATION_TYPE_INDUSTRY_TYPE_8', 'ORGANIZATION_TYPE_INDUSTRY_TYPE_9', 'ORGANIZATION_TYPE_INSURANCE',
 'ORGANIZATION_TYPE_KINDERGARTEN', 'ORGANIZATION_TYPE_LEGAL_SERVICES', 'ORGANIZATION_TYPE_MEDICINE',
 'ORGANIZATION_TYPE_MILITARY', 'ORGANIZATION_TYPE_MOBILE', 'ORGANIZATION_TYPE_OTHER',
 'ORGANIZATION_TYPE_POLICE', 'ORGANIZATION_TYPE_POSTAL', 'ORGANIZATION_TYPE_REALTOR',
 'ORGANIZATION_TYPE_RELIGION', 'ORGANIZATION_TYPE_RESTAURANT', 'ORGANIZATION_TYPE_SCHOOL',
 'ORGANIZATION_TYPE_SECURITY', 'ORGANIZATION_TYPE_SECURITY_MINISTRIES', 'ORGANIZATION_TYPE_SELF_EMPLOYED',
 'ORGANIZATION_TYPE_SERVICES', 'ORGANIZATION_TYPE_TELECOM', 'ORGANIZATION_TYPE_TRADE_TYPE_1',
 'ORGANIZATION_TYPE_TRADE_TYPE_2', 'ORGANIZATION_TYPE_TRADE_TYPE_3', 'ORGANIZATION_TYPE_TRADE_TYPE_4',
 'ORGANIZATION_TYPE_TRADE_TYPE_5', 'ORGANIZATION_TYPE_TRADE_TYPE_6', 'ORGANIZATION_TYPE_TRADE_TYPE_7',
 'ORGANIZATION_TYPE_TRANSPORT_TYPE_1', 'ORGANIZATION_TYPE_TRANSPORT_TYPE_2',
 'ORGANIZATION_TYPE_TRANSPORT_TYPE_3', 'ORGANIZATION_TYPE_TRANSPORT_TYPE_4', 'ORGANIZATION_TYPE_UNIVERSITY',
 'ORGANIZATION_TYPE_XNA', 'ORGANIZATION_TYPE_NAN', 'FONDKAPREMONT_MODE_NOT_SPECIFIED',
 'FONDKAPREMONT_MODE_ORG_SPEC_ACCOUNT', 'FONDKAPREMONT_MODE_REG_OPER_ACCOUNT',
 'FONDKAPREMONT_MODE_REG_OPER_SPEC_ACCOUNT', 'FONDKAPREMONT_MODE_NAN', 'HOUSETYPE_MODE_BLOCK_OF_FLATS',
 'HOUSETYPE_MODE_SPECIFIC_HOUSING', 'HOUSETYPE_MODE_TERRACED_HOUSE', 'HOUSETYPE_MODE_NAN',
 'WALLSMATERIAL_MODE_BLOCK', 'WALLSMATERIAL_MODE_MIXED', 'WALLSMATERIAL_MODE_MONOLITHIC',
 'WALLSMATERIAL_MODE_OTHERS', 'WALLSMATERIAL_MODE_PANEL', 'WALLSMATERIAL_MODE_STONE_BRICK',
 'WALLSMATERIAL_MODE_WOODEN', 'WALLSMATERIAL_MODE_NAN', 'EMERGENCYSTATE_MODE_NO', 'EMERGENCYSTATE_MODE_YES',
 'EMERGENCYSTATE_MODE_NAN']

FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS_ALL if not c.startswith(EXCLUDE_PREFIXES)]

USECOLS = {
    "bureau.csv": [ID, BUREAU_ID, "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
                   "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                   "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE"],
    "bureau_balance.csv": [BUREAU_ID, "MONTHS_BALANCE", "STATUS"],
    "previous_application.csv": [ID, "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO",
                                 "NAME_YIELD_GROUP", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
                                 "AMT_GOODS_PRICE", "RATE_DOWN_PAYMENT", "CNT_PAYMENT", "DAYS_DECISION"],
    "POS_CASH_balance.csv": [ID, "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS",
                             "SK_DPD", "SK_DPD_DEF"],
    "installments_payments.csv": [ID, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
    "credit_card_balance.csv": [ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
                                "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "CNT_DRAWINGS_CURRENT",
                                "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"],
}

# Funciones Generales

def clean_name(text: str) -> str:
    """
        Convierte un texto en un nombre de columna valido
        (ej. 'Consumer Loans' -> 'CONSUMER_LOANS')
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").upper()

def read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Lee el csv, y reduce el tamaño de las variables cuando es
        posible para ahorrar memoria."""

    df = pd.read_csv(path, usecols=usecols, dtype={c: str for c in usecols if c in TEXT_COLS})
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator != 0)
    return result.replace([np.inf, -np.inf], np.nan)

def aggregate_numeric(df: pd.DataFrame, spec: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    """
        df: Es la tabla que se esta trabajando
        spec: {columna: [estadísticos]} Nos dice de cada columna a procesar de la tabla
        cuales estadísticos vamos a calcular.
        prefix: prefijo para identificar de que tabla viene la variable. (ej. BUREAU_)
    """

    agg = df.groupby(ID).agg(spec)
    agg.columns = [f"{prefix}{col}_{stat.upper()}" for col, stat in agg.columns]
    agg[f"{prefix}COUNT"] = df.groupby(ID).size()
    return agg

def categorical_shares(df: pd.DataFrame, cols: list[str],  prefix: str) -> pd.DataFrame:
    """
        Convierte las variables en proporciones por cliente.
        Agrega las columnas de categorias que no tiene el cliente para que esas solo
        queden como proporción 0 y no ausentes.
    """
    dummies = pd.get_dummies(df[cols], dtype="float32")
    dummies.columns = [f"{prefix}{clean_name(c)}_SHARE" for c in dummies.columns]
    dummies[ID] = df[ID].values
    shares = dummies.groupby(ID).mean()
    want = [c for c in FEATURE_COLUMNS if c.endswith("_SHARE") and any(c.startswith(f"{prefix}{clean_name(col)}_") for col in cols)]
    return shares.reindex(columns=want, fill_value=0.0)

def recent_trend_features(
        df: pd.DataFrame,
        time_col: str,
        value_cols: list[str],
        windows: list[int],
        prefix: str
) -> pd.DataFrame:
    """
        Define las tendencias recientes de comportamiento a comparación con el historial completo.
    """
    # Prmedio historico de todos los datos del cliente.
    full_mean = df.groupby(ID)[value_cols].mean()

    blocks = []
    for w in windows:
        # Toma los datos que esten en las ventanas de tiempo.
        recent = df[df[time_col] >= -w]

        # Promedio de los datos en la ventana de tiempo.
        recent_mean = recent.groupby(ID)[value_cols].mean().reindex(full_mean.index)


        mean_block = recent_mean.copy()
        mean_block.columns = [f"{prefix}{c}_MEAN_L{w}" for c in value_cols]

        diff_block = recent_mean - full_mean
        diff_block.columns = [f"{prefix}{c}_DIFF_L{w}" for c in value_cols]

        ratio_block = safe_divide(recent_mean, full_mean)
        ratio_block.columns = [f"{prefix}{c}_RATIO_L{w}" for c in value_cols]

        blocks.extend([mean_block, diff_block, ratio_block])

    return pd.concat(blocks, axis=1)

def process_previous_application(data_dir: Path) -> pd.DataFrame:
    """
        Procesa previous_application.csv.
    """
    cols = [
        ID, "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT",
        "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "CNT_PAYMENT",
        "DAYS_DECISION", "RATE_DOWN_PAYMENT",
        "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE",
        "NAME_PORTFOLIO", "CODE_REJECT_REASON",
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE", "DAYS_TERMINATION",
    ]

    df = read_csv(data_dir / "previous_application.csv", cols)

    """
        Esta anomalía sucede cuando el valor en columnas con DAYS_ tiene un valor de
        365,243 (~1000 años). Esto representa un no aplica o no ha ocurrido, no un
        dato real.
        Si se deja esto se distorsionaría las variables de agregación por eso mejor
        se cambia a un np.nan.
    """
    days_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                 "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    df[days_cols] = df[days_cols].replace(365243, np.nan)

    # Variable que nos dice cuánto credito se le aprobó vs. cuánto pidió.
    # Un ratio < 1 dice que el banco lo considero riesgoso y le dio menos de lo que solicito.
    df["APP_CREDIT_RATIO"] = safe_divide(df["AMT_APPLICATION"], df["AMT_CREDIT"])

    spec = {
        "AMT_ANNUITY": STATS, # Monto de la anualidad del préstamo.
        "AMT_APPLICATION": STATS, # Monto del crédito solicitado previamente.
        "AMT_CREDIT": STATS, # Monto del credito del préstamo.
        "AMT_DOWN_PAYMENT": STATS, # Monto del pago inicial.
        "AMT_GOODS_PRICE": STATS, # Precio de los bienes financiados (si aplica).
        "CNT_PAYMENT": ["mean", "max", "sum"], # Plazo del credito previo.
        "DAYS_DECISION": ["mean", "max", "min"], # Dias respecto a la decisión previa.
        "RATE_DOWN_PAYMENT": ["mean", "max"], # Tasa o interes normalizado.
        "DAYS_FIRST_DRAWING": ["min", "max"], # Días desde el primer desembolso.
        "DAYS_FIRST_DUE": ["min", "max"], # Días desde la primera fecha de vencimiento.
        "DAYS_LAST_DUE_1ST_VERSION": ["min", "max"], # Días desde la primera versión de la
                                                     # ultima fecha de vencimiento.
        "DAYS_LAST_DUE": ["min", "max"], # Días desde la ultima fecha de vencimiento.
        "DAYS_TERMINATION": ["min", "max"], # Días esperados de terminación.
        "APP_CREDIT_RATIO": ["mean", "min", "max"], # Variable creada anteriormente.
    }

    features = aggregate_numeric(df, spec, "PREV_")

    shares = categorical_shares(
        df,
        ["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE",
         "NAME_PORTFOLIO", "CODE_REJECT_REASON"],
        "PREV_"
    )

    trend = recent_trend_features(
        df, "DAYS_DECISION",
        ["AMT_CREDIT", "APP_CREDIT_RATIO"],
        WINDOWS_DAYS, "PREV_"
    )

    for w in WINDOWS_DAYS:
        n_recent = df[df["DAYS_DECISION"] >= -w].groupby(ID).size()
        trend[f"PREV_N_APPLICATIONS_L{w}"] = n_recent.reindex(trend.index).fillna(0)

    return features.join(shares, how="left").join(trend, how="left")

def process_bureau(data_dir: Path) -> pd.DataFrame:
    """
        Procesa bureau.csv.
    """
    cols = [
        ID, "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
        "CREDIT_ACTIVE", "CREDIT_TYPE",
    ]
    df = read_csv(data_dir / "bureau.csv", cols)

    # Fracción del crédito que aún se debe.
    df["DEBT_CREDIT_RATIO"] = safe_divide(df["AMT_CREDIT_SUM_DEBT"], df["AMT_CREDIT_SUM"])

    spec = {
        "DAYS_CREDIT": ["min", "max", "mean"], # Días desde la solicitud en el buro.
        "CREDIT_DAY_OVERDUE": STATS, # Días de atraso en el credito del buro.
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"], # Duración del credito en el buro.
        "AMT_CREDIT_SUM": STATS, # Monto actual de credito en el buro.
        "AMT_CREDIT_SUM_DEBT": STATS, # Deuda actual enel credito del buro.
        "AMT_CREDIT_SUM_OVERDUE": STATS, # Monto vencido actual en el credito del buro.
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"], # Monto maximo vencido en el credito.
        "CNT_CREDIT_PROLONG": ["sum", "max"], # Número de prolongaciones del crédito en el buro.
        "DEBT_CREDIT_RATIO": ["max", "mean"], # Variable agregada.
    }

    features = aggregate_numeric(df, spec, "BUREAU_")

    shares = categorical_shares(df, ["CREDIT_ACTIVE", "CREDIT_TYPE"], "BUREAU_")

    trend = recent_trend_features(
        df, "DAYS_CREDIT",
        ["CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"],
        WINDOWS_DAYS, "BUREAU_",
    )

    return features.join(shares, how="left").join(trend, how="left")

def process_bureau_balance(data_dir: Path) -> pd.DataFrame:
    """
        Procesa bureau_balance.csv.
    """

    bb = read_csv(data_dir / "bureau_balance.csv", [BUREAU_ID, "MONTHS_BALANCE", "STATUS"])
    id_map = read_csv(data_dir / "bureau.csv", [ID, BUREAU_ID]).drop_duplicates(BUREAU_ID)
    bb = bb.merge(id_map, on=BUREAU_ID, how="inner")

    # Escala númerica de severidad de mora
    BB_SEVERITY = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    # Mapeado del estatus a su valor numerico (SEVERITY) y bandera (DELINQUENT) para saber si hubo mora de cualquier tipo.
    bb["SEVERITY"] = bb["STATUS"].map(BB_SEVERITY).astype("float32")
    bb["DELINQUENT"] = (bb["SEVERITY"] >= 1).astype("float32")
    bb.loc[bb["SEVERITY"].isna(), "DELINQUENT"] = np.nan

    spec = {
        "MONTHS_BALANCE": ["min", "max"], # Longitud del historial en el buro.
        "SEVERITY": ["mean", "max", "std", "sum"],
        "DELINQUENT": ["mean", "sum"],
    }

    features = aggregate_numeric(bb, spec, "BB_")

    shares = categorical_shares(bb, ["STATUS"], "BB_")

    loan = bb.groupby(BUREAU_ID).agg(client=(ID, "first"), worst=("SEVERITY", "max"))
    by_client = loan.groupby("client")["worst"]
    loan_feats = pd.DataFrame({
        "BB_LOAN_WORST_SEVERITY_MEAN": by_client.mean(),
        "BB_LOAN_WORST_SEVERITY_MAX": by_client.max(),
        "BB_LOAN_EVER_DELINQUENT_SHARE": loan["worst"].ge(1).groupby(loan["client"]).mean(),
    }).rename_axis(ID)

    trend = recent_trend_features(
        bb, "MONTHS_BALANCE",
        ["SEVERITY", "DELINQUENT"],
        WINDOWS_MONTHS, "BB_"
    )

    return features.join(shares, how="left").join(loan_feats, how="left").join(trend, how="left")


def process_installments(data_dir: Path) -> pd.DataFrame:
    """
        Procesa instalLments_payments.csv.
    """

    cols = [
        ID, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT", "AMT_PAYMENT",
    ]
    df = read_csv(data_dir / "installments_payments.csv", cols)

    # Días de retraso (DPD = Days Past Due).
    delta = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["DPD"] = delta.clip(lower=0)

    # Que tan anticipado paga (DBD = Days Before Due).
    df["DBD"] = (-delta).clip(lower=0)

    # Bandera 0/1 de pago tardio.
    df["LATE_PAYMENT"] = (df["DPD"] > 0).astype("float32")
    df.loc[df["DPD"].isna(), "LATE_PAYMENT"] = np.nan

    # Cuota pagada vs. cuota exigida.
    df["AMT_PAYMENT"] = df["AMT_PAYMENT"].fillna(0) # Se llena con cero porque si no se tiene información se toma como que no realizo el pago.
    df["PAYMENT_RATIO"] = safe_divide(df["AMT_PAYMENT"], df["AMT_INSTALMENT"])

    # Diferencia de cuota pagada y cuota exigida (Saber si pago más o pago menos).
    df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    spec = {
        "DPD": STATS,
        "DBD": STATS,
        "LATE_PAYMENT": ["mean", "sum"],
        "AMT_INSTALMENT": ["sum", "mean", "max"], # Pago requerido
        "AMT_PAYMENT": ["sum", "mean", "max"], # Pago realizado
        "PAYMENT_RATIO": ["mean", "min", "std"],
        "PAYMENT_DIFF": STATS,
    }

    features = aggregate_numeric(df, spec, "INST_")

    trend = recent_trend_features(
        df, "DAYS_INSTALMENT",
        ["DPD", "LATE_PAYMENT", "PAYMENT_RATIO", "PAYMENT_DIFF"],
        WINDOWS_DAYS, "INST_",
    )

    return features.join(trend, how="left")

def process_pos_cash(data_dir: Path) -> pd.DataFrame:
    """
        Procesa POS_CASH_balance.csv.
    """
    cols = [
        ID, "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
        "SK_DPD", "SK_DPD_DEF", "NAME_CONTRACT_STATUS",
    ]
    df = read_csv(data_dir / "POS_CASH_balance.csv", cols)

    # Bandera de mes con atraso
    df["DPD_FLAG"] = (df["SK_DPD"] > 0).astype("float32")

    spec = {
        "MONTHS_BALANCE": ["min", "max"], # Longitud del historial
        "SK_DPD": STATS, # Días de atraso
        "SK_DPD_DEF": STATS, # Atraso "tolerado"
        "DPD_FLAG": ["mean", "sum"],
        "CNT_INSTALMENT": ["mean", "max"], # Plazo del credito
        "CNT_INSTALMENT_FUTURE": ["mean", "min"], # Cuotas que faltan por pagar
    }

    features = aggregate_numeric(df, spec, "POS_")

    shares = categorical_shares(df, ["NAME_CONTRACT_STATUS"], "POS_")

    trend = recent_trend_features(
        df, "MONTHS_BALANCE",
        ["SK_DPD", "SK_DPD_DEF", "DPD_FLAG"],
        WINDOWS_MONTHS, "POS_",
    )

    return features.join(shares, how="left").join(trend, how="left")


def process_credit_card(data_dir: Path) -> pd.DataFrame:
    """
        Procesa credit_card_balance.csv.
    """
    cols = [
        ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_TOTAL_RECEIVABLE", "SK_DPD",
        "SK_DPD_DEF", "NAME_CONTRACT_STATUS",
    ]
    df = read_csv(data_dir / "credit_card_balance.csv", cols)

    # Tasa de utilización (Saldo / Limite)
    df["UTILIZATION"] = safe_divide(df["AMT_BALANCE"], df["AMT_CREDIT_LIMIT_ACTUAL"])

    # Pago ralizado vs. pago minimo exigido.
    df["PAYMENT_MIN_RATIO"] = safe_divide(df["AMT_PAYMENT_CURRENT"], df["AMT_INST_MIN_REGULARITY"])

    spec = {
        "MONTHS_BALANCE": ["min", "max"], # Longitud del historial
        "AMT_BALANCE": STATS, # Saldo durante el mes
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"], # Limite de tarjeta de credito durante el mes
        "AMT_DRAWINGS_CURRENT": STATS, # Monto de disposiciones o retiros
        "AMT_PAYMENT_CURRENT": STATS,  # Monto de pago
        "AMT_TOTAL_RECEIVABLE": ["mean", "max"], # Monto por cobrar
        "SK_DPD": STATS, # Días de atraso
        "SK_DPD_DEF": STATS, # Días de atraso en estado de default
        "UTILIZATION": ["mean", "max", "min", "std"],
        "PAYMENT_MIN_RATIO": ["mean", "min"],
    }

    features = aggregate_numeric(df, spec, "CC_")

    shares = categorical_shares(df, ["NAME_CONTRACT_STATUS"], "CC_")

    trend = recent_trend_features(
        df, "MONTHS_BALANCE",
        ["UTILIZATION", "AMT_BALANCE", "SK_DPD", "PAYMENT_MIN_RATIO"],
        WINDOWS_MONTHS, "CC_",
    )

    return features.join(shares, how="left").join(trend,how="left")

def add_leverage_ratios(app: pd.DataFrame) -> pd.DataFrame:
    """
        Agrega los ratios de apalancamiento calculados sobre la solitud actual.
    """
    app = app.copy()
    app["DEBT_TO_INCOME"] = safe_divide(app["AMT_ANNUITY"], app["AMT_INCOME_TOTAL"])
    app["CREDIT_TO_GOODS"] = safe_divide(app["AMT_CREDIT"], app["AMT_GOODS_PRICE"])
    app["CREDIT_TO_INCOME"] = safe_divide(app["AMT_CREDIT"], app["AMT_INCOME_TOTAL"])
    app["ANNUITY_TO_CREDIT"] = safe_divide(app["AMT_ANNUITY"], app["AMT_CREDIT"])
    return app

def process_application(app: pd.DataFrame) -> pd.DataFrame:

    """
        Procesa la tabla de aplicaciones principal para corregir anomalias y agregar ratios o datos utiles.
    """
    app = app.copy()

    # Anomalia conocida
    app["DAYS_EMPLOYED_ANOM"] = (app["DAYS_EMPLOYED"] == 365243).astype("int8")
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
    if "CODE_GENDER" in app.columns:
        app["CODE_GENDER"] = app["CODE_GENDER"].replace("XNA", np.nan)

    # Ratios de apalancamiento
    app = add_leverage_ratios(app)

    # Ratios demográficos / de ingreso
    app["DAYS_EMPLOYED_PERC"] = safe_divide(app["DAYS_EMPLOYED"], app["DAYS_BIRTH"]) # Fracción de vida que lleva en su empleo actual.
    app["INCOME_PER_PERSON"] = safe_divide(app["AMT_INCOME_TOTAL"], app["CNT_FAM_MEMBERS"]) # Ingreso disponible por integrante en la familia.

    # Resumen de scores externos
    ext = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    app["EXT_SOURCES_MEAN"] = ext.mean(axis=1)
    app["EXT_SOURCES_MIN"] = ext.min(axis=1)
    app["EXT_SOURCES_MAX"] = ext.max(axis=1)
    app["EXT_SOURCES_STD"] = ext.std(axis=1)

    # Cantidad de documentos entregados
    doc_cols = [c for c in app.columns if c.startswith("FLAG_DOCUMENT_")]
    app["DOCUMENTS_PROVIDED"] = app[doc_cols].sum(axis=1)

    return app

def build_feature_matrix(app_path: Path, tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
        Une las tablas procesadas en una sola y con la tablas de train y test.
    """
    app = pd.read_csv(app_path)
    app =  process_application(app).set_index(ID)

    # Las columnas no colisionan ya que cada tabla usa su propio prefijo.
    features = app.join(tables, how="left").copy()

    # Remplaza NaN por cero ya que NaN en estos registros significa que no hay registros no que no hay datos.
    count_cols = [c for c in features.columns if c.endswith("_COUNT")]
    features[count_cols] = features[count_cols].fillna(0)

    return features.reset_index()



def list_partitions(part_dir: Path) -> list[tuple[int, Path]]:
    """
        Regresa lalista de particiones terminadas en data/partitioned/K*/.
    """
    found = []
    for kdir in sorted(part_dir.glob("K*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        if kdir.is_dir() and kdir.name[1:].isdigit():
            found += [(int(kdir.name[1:]), d) for d in sorted(kdir.iterdir()) if d.is_dir() and (d / "_COMPLETE").exists()]

    return found

def encode_application(app: pd.DataFrame) -> pd.DataFrame:
    """
        Aqui se asegura que el cliente tenga las columnas categoricas aun esten vacias.
    """

    app = app.copy()
    for c in APP_CAT_COLS:
        app[c] = app[c].astype(object).where(app[c].notna(), np.nan)

    original_cols = set(app.columns)
    app = pd.get_dummies(app, columns=APP_CAT_COLS, dummy_na=True, dtype="float32")

    return app.rename(columns={c: clean_name(c) for c in app.columns if c not in original_cols})



def build_client(cdir: Path) -> pd.DataFrame:
    """
        Regresa la matriz de variables de un cliente a partir de sus tablas crudas.
        Devuelve el dataframe con la matriz lista para el entrenamiento.
    """

    tables = []
    for label, fn in [
        ("previous_application", process_previous_application),
        ("bureau", process_bureau),
        ("bureau_balance", process_bureau_balance),
        ("installments_payments", process_installments),
        ("POS_CASH_balance", process_pos_cash),
        ("credit_card_balance", process_credit_card),
    ]:
        feats = fn(cdir)
        tables.append(feats)
        print(f"{label}: {feats.shape[0]:,} filas x {feats.shape[1]:,} variables")

    x = encode_application(build_feature_matrix(cdir / "application_train.csv", tables))
    x = x.reindex(columns=FEATURE_COLUMNS)
    dummy_cols = [c for c in FEATURE_COLUMNS if any(c.startswith(clean_name(COL) + "_") for COL in APP_CAT_COLS)]
    x[dummy_cols] = x[dummy_cols].fillna(0.0)
    count_cols = [c for c in FEATURE_COLUMNS if c.endswith("_COUNT")]
    x[count_cols] = x[count_cols].fillna(0)
    return x


def process_client(args: tuple) -> tuple[int, int, list[str]]:
    """
        Procesa un cliente y escribe el parquet.
    """
    j, cdir, out_file = args
    x = build_client(cdir).set_index(ID)
    split = pd.read_csv(cdir / "split.csv").set_index(ID)
    feat_cols = [c for c in FEATURE_COLUMNS if c not in (ID, TARGET)]
    x = x.astype({c: "float32" for c in feat_cols})
    out = pd.concat([x[[TARGET]], split.loc[x.index, ["global_split", "local_split"]], x[feat_cols]], axis=1).copy().reset_index()
    out.to_parquet(out_file, index=False)
    return j, len(out), len(feat_cols)

def main():
    setup_console()

    parser = argparse.ArgumentParser(description="Procesamiento de datos por cliente.")
    add_sample_arg(parser)
    parser.add_argument("--partitions", default=None,
                        help="Lista separada por comas. Sin esto, procesa todas las completas.")
    args = parser.parse_args()

    paths = get_paths(args.sample)
    paths.make()
    part_dir, out_root = paths.partitioned, paths.federated
    parts = list_partitions(part_dir)

    if args.partitions:
        wanted = set(args.partitions.split(","))
        parts = [(K, d) for K, d in parts if f"K{K}_{d.name}" in wanted]

    if not parts:
        raise SystemExit(f"No hay particiones compoletas en {part_dir}. Ejecuta primero el script partition_data.py.")

    pending = [(K, d) for K , d in parts if not (out_root / f"K{K}" / d.name / "_PROCESSED").exists()]
    print(f"Particiones: {len(parts)} ({len(parts)-len(pending)} ya procesadas, {len(pending)} por procesar) | Salida: {out_root}")

    for n, (K, d) in enumerate(pending, 1):
        t0 = time.time()
        out_dir = out_root / f"K{K}" / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(j, cd, out_dir / f"client_{j:02d}.parquet") for j, cd in enumerate(sorted(p for p in d.glob("client_*") if p.is_dir()))]
        if N_WORKERS > 1:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                res = list(pool.map(process_client, jobs))
        else:
            res = [process_client(j) for j in jobs]

        print(f"[{n}/{len(pending)}] K={K:<3} {d.name:<28} {sum(r[1] for r in res):>8} solicitantes x {res[0][2]} variables ({time.time() - t0:5.1f}s)")

        (out_dir / "_PROCESSED").write_text("ok\n")

    print("Listo. Clientes procesados.")

if __name__ == "__main__":
    main()