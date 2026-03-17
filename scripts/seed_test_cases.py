import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings

try:
    import aiomysql
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("aiomysql is required to seed MySQL test data.") from exc


def build_sample_cases() -> List[Dict[str, object]]:
    return [
        {
            "case_id": "CASE-0001",
            "reported_persons": ["王建国"],
            "reporter": "张某",
            "location": "太原市",
            "location_district": "万柏林区",
            "description_text": "反映万柏林区住建局副局长王建国在城中村改造项目中收受工程老板礼金，并指定亲属承包附属工程。",
            "create_time": "2024-01-12 10:00:00",
            "updated_at": "2024-01-12 10:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["工程建设", "亲属承揽"]},
        },
        {
            "case_id": "CASE-0002",
            "reported_persons": ["王建国"],
            "reporter": "李某",
            "location": "太原市",
            "location_district": "万柏林区",
            "description_text": "举报王建国利用城改安置房配套工程谋私，收受承包人购物卡和现金，并让外甥参与施工。",
            "create_time": "2024-03-02 09:30:00",
            "updated_at": "2024-03-02 09:30:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["重复件候选", "礼金"]},
        },
        {
            "case_id": "CASE-0003",
            "reported_persons": ["王建国"],
            "reporter": "匿名",
            "location": "太原市",
            "location_district": "万柏林区",
            "description_text": "反映王建国在老旧小区改造中违规接受请托，除礼金外还安排姐姐名下公司承揽围挡工程。",
            "create_time": "2024-05-18 14:20:00",
            "updated_at": "2024-05-18 14:20:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["新线索", "亲属公司"]},
        },
        {
            "case_id": "CASE-0004",
            "reported_persons": ["李海峰"],
            "reporter": "王某",
            "location": "太原市",
            "location_district": "小店区",
            "description_text": "反映小店区街道干部李海峰在征地补偿中优亲厚友，冒领补偿款。",
            "create_time": "2023-11-01 08:10:00",
            "updated_at": "2023-11-01 08:10:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["征地补偿"]},
        },
        {
            "case_id": "CASE-0005",
            "reported_persons": ["李海峰"],
            "reporter": "赵某",
            "location": "太原市",
            "location_district": "小店区",
            "description_text": "举报李海峰在城中村拆迁过程中为亲属多报面积，套取补偿资金。",
            "create_time": "2024-02-21 16:00:00",
            "updated_at": "2024-02-21 16:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["同人不同表述"]},
        },
        {
            "case_id": "CASE-0006",
            "reported_persons": ["赵志强"],
            "reporter": "学校家长代表",
            "location": "大同市",
            "location_district": "平城区",
            "description_text": "反映平城区教育局采购办赵志强指定某公司中标校服采购并收受回扣。",
            "create_time": "2024-04-03 11:20:00",
            "updated_at": "2024-04-03 11:20:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["教育采购"]},
        },
        {
            "case_id": "CASE-0007",
            "reported_persons": ["赵志强"],
            "reporter": "刘某",
            "location": "大同市",
            "location_district": "平城区",
            "description_text": "举报赵志强在校服和课桌采购中偏袒固定供应商，收取好处费。",
            "create_time": "2024-07-15 13:05:00",
            "updated_at": "2024-07-15 13:05:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["采购回扣"]},
        },
        {
            "case_id": "CASE-0008",
            "reported_persons": ["陈国平"],
            "reporter": "货运司机",
            "location": "运城市",
            "location_district": "盐湖区",
            "description_text": "反映盐湖区交通执法人员陈国平对煤运车辆违规罚款，并私下收钱放行。",
            "create_time": "2023-09-09 12:00:00",
            "updated_at": "2023-09-09 12:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["执法乱收费"]},
        },
        {
            "case_id": "CASE-0009",
            "reported_persons": ["陈国平"],
            "reporter": "匿名",
            "location": "运城市",
            "location_district": "盐湖区",
            "description_text": "举报陈国平在超限检查站收受司机现金后不录入处罚系统，涉嫌吃拿卡要。",
            "create_time": "2024-01-27 17:45:00",
            "updated_at": "2024-01-27 17:45:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["吃拿卡要"]},
        },
        {
            "case_id": "CASE-0010",
            "reported_persons": ["刘春生"],
            "reporter": "村民代表",
            "location": "临汾市",
            "location_district": "尧都区",
            "description_text": "反映乡镇干部刘春生在扶贫项目验收中收礼后放宽标准，相关资金去向不明。",
            "create_time": "2022-12-05 10:30:00",
            "updated_at": "2022-12-05 10:30:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["扶贫资金"]},
        },
        {
            "case_id": "CASE-0011",
            "reported_persons": ["刘春生"],
            "reporter": "韩某",
            "location": "临汾市",
            "location_district": "尧都区",
            "description_text": "举报刘春生在产业帮扶项目验收时收受烟酒和现金，并虚报完成进度。",
            "create_time": "2024-06-11 09:00:00",
            "updated_at": "2024-06-11 09:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["项目验收"]},
        },
        {
            "case_id": "CASE-0012",
            "reported_persons": ["孙志明"],
            "reporter": "干部群众",
            "location": "晋中市",
            "location_district": "榆次区",
            "description_text": "反映榆次区干部考察期间，孙志明收受礼金帮助他人调岗晋升。",
            "create_time": "2024-03-09 15:00:00",
            "updated_at": "2024-03-09 15:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["选人用人"]},
        },
        {
            "case_id": "CASE-0013",
            "reported_persons": ["孙志明"],
            "reporter": "实名举报人",
            "location": "晋中市",
            "location_district": "榆次区",
            "description_text": "举报孙志明在干部调整中收礼打招呼，帮助关系人进入重要岗位。",
            "create_time": "2024-04-16 10:10:00",
            "updated_at": "2024-04-16 10:10:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["岗位调整"]},
        },
        {
            "case_id": "CASE-0014",
            "reported_persons": ["郭晓军"],
            "reporter": "工程监理",
            "location": "忻州市",
            "location_district": "忻府区",
            "description_text": "反映郭晓军在市政道路改造中虚增工程量，为承包商套取工程款。",
            "create_time": "2023-10-12 11:30:00",
            "updated_at": "2023-10-12 11:30:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["虚增工程量"]},
        },
        {
            "case_id": "CASE-0015",
            "reported_persons": ["郭晓军"],
            "reporter": "匿名",
            "location": "忻州市",
            "location_district": "忻府区",
            "description_text": "举报郭晓军与施工单位串通，在道路提升项目中做假签证、套取资金。",
            "create_time": "2024-02-02 18:20:00",
            "updated_at": "2024-02-02 18:20:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["工程签证"]},
        },
        {
            "case_id": "CASE-0016",
            "reported_persons": ["马文斌"],
            "reporter": "医院职工",
            "location": "长治市",
            "location_district": "潞州区",
            "description_text": "反映某医院设备科负责人马文斌在影像设备采购中收受供应商回扣。",
            "create_time": "2024-01-04 08:40:00",
            "updated_at": "2024-01-04 08:40:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["医疗采购"]},
        },
        {
            "case_id": "CASE-0017",
            "reported_persons": ["马文斌"],
            "reporter": "知情人",
            "location": "长治市",
            "location_district": "潞州区",
            "description_text": "举报马文斌在CT维保和试剂采购中指定品牌，存在返点问题。",
            "create_time": "2024-05-28 13:15:00",
            "updated_at": "2024-05-28 13:15:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["设备维保"]},
        },
        {
            "case_id": "CASE-0018",
            "reported_persons": ["高瑞红"],
            "reporter": "企业员工",
            "location": "吕梁市",
            "location_district": "离石区",
            "description_text": "反映高瑞红利用国企改制之机低价入股关联企业，涉嫌利益输送。",
            "create_time": "2023-08-19 14:00:00",
            "updated_at": "2023-08-19 14:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["国企改制"]},
        },
        {
            "case_id": "CASE-0019",
            "reported_persons": ["高瑞红"],
            "reporter": "杨某",
            "location": "吕梁市",
            "location_district": "离石区",
            "description_text": "举报高瑞红操纵关联公司承接改制资产，并以代持方式隐匿股份。",
            "create_time": "2024-02-09 10:55:00",
            "updated_at": "2024-02-09 10:55:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["代持股份"]},
        },
        {
            "case_id": "CASE-0020",
            "reported_persons": ["张明亮"],
            "reporter": "村干部",
            "location": "朔州市",
            "location_district": "朔城区",
            "description_text": "反映张明亮在土地征收中压低补偿标准，对亲属额外照顾。",
            "create_time": "2023-07-07 10:00:00",
            "updated_at": "2023-07-07 10:00:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["土地征收"]},
        },
        {
            "case_id": "CASE-0021",
            "reported_persons": ["张明亮"],
            "reporter": "匿名",
            "location": "朔州市",
            "location_district": "朔城区",
            "description_text": "举报张明亮在征地补偿审核中对关系户多发补偿款，普通村民反而被压减。",
            "create_time": "2024-01-30 09:10:00",
            "updated_at": "2024-01-30 09:10:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["审核偏私"]},
        },
        {
            "case_id": "CASE-0022",
            "reported_persons": ["郝志军"],
            "reporter": "群众",
            "location": "晋城市",
            "location_district": "城区",
            "description_text": "反映郝志军在棚户区安置房分配中违规插手房源，优先照顾亲友。",
            "create_time": "2024-03-21 16:35:00",
            "updated_at": "2024-03-21 16:35:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["安置房分配"]},
        },
        {
            "case_id": "CASE-0023",
            "reported_persons": ["梁晓峰"],
            "reporter": "匿名",
            "location": "阳泉市",
            "location_district": "城区",
            "description_text": "举报梁晓峰在环保执法中对矿企通风报信，导致检查流于形式。",
            "create_time": "2024-04-01 14:45:00",
            "updated_at": "2024-04-01 14:45:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["环保执法"]},
        },
        {
            "case_id": "CASE-0024",
            "reported_persons": ["程立新"],
            "reporter": "刘某",
            "location": "太原市",
            "location_district": "尖草坪区",
            "description_text": "反映程立新在危房改造补助审核中拖延审批，但未见明确收受财物线索。",
            "create_time": "2024-06-20 12:30:00",
            "updated_at": "2024-06-20 12:30:00",
            "status": "ACTIVE",
            "extra_json": {"tags": ["非重复对照"]},
        },
    ]


async def seed_cases(args) -> None:
    settings = get_settings()
    connection = await aiomysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        db=settings.mysql_db,
        autocommit=True,
        charset="utf8mb4",
    )
    try:
        async with connection.cursor() as cursor:
            if args.create_schema:
                schema_sql = (
                    PROJECT_ROOT / "scripts" / "create_test_schema.sql"
                ).read_text(encoding="utf-8")
                for statement in schema_sql.split(";"):
                    normalized = statement.strip()
                    if not normalized:
                        continue
                    await cursor.execute(normalized)
            if args.truncate:
                await cursor.execute(
                    "TRUNCATE TABLE {0}".format(settings.mysql_source_table)
                )
                await cursor.execute(
                    "TRUNCATE TABLE {0}".format(settings.mysql_wtxx_table)
                )
                await cursor.execute(
                    "TRUNCATE TABLE {0}".format(settings.mysql_xfj_table)
                )

            legacy_rows = []
            xfj_rows = []
            wtxx_rows = []
            for item in build_sample_cases():
                petition_id = "XFJ-{0}".format(item["case_id"])
                legacy_rows.append(
                    (
                        item["case_id"],
                        json.dumps(item["reported_persons"], ensure_ascii=False),
                        item["reporter"],
                        item["location"],
                        item["location_district"],
                        item["description_text"],
                        item["create_time"],
                        item["updated_at"],
                        item["status"],
                        json.dumps(item["extra_json"], ensure_ascii=False),
                    )
                )
                xfj_rows.append(
                    (
                        petition_id,
                        ",".join(item["reported_persons"]),
                        item["reporter"],
                        item["location"],
                        item["create_time"],
                        item["updated_at"],
                    )
                )
                wtxx_rows.append(
                    (
                        item["case_id"],
                        petition_id,
                        item["description_text"].encode("utf-8"),
                        item["create_time"],
                        item["updated_at"],
                    )
                )
            await cursor.executemany(
                """
                INSERT INTO {source_table} (
                    case_id,
                    reported_persons_json,
                    reporter,
                    location,
                    location_district,
                    description_text,
                    create_time,
                    updated_at,
                    status,
                    extra_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) AS incoming
                ON DUPLICATE KEY UPDATE
                    reported_persons_json = incoming.reported_persons_json,
                    reporter = incoming.reporter,
                    location = incoming.location,
                    location_district = incoming.location_district,
                    description_text = incoming.description_text,
                    create_time = incoming.create_time,
                    updated_at = incoming.updated_at,
                    status = incoming.status,
                    extra_json = incoming.extra_json
                """.format(source_table=settings.mysql_source_table),
                legacy_rows,
            )
            await cursor.executemany(
                """
                INSERT INTO {xfj_table} (
                    C_BH,
                    C_BFYR_XX,
                    C_FYR_XX,
                    C_WTSD_QC,
                    DT_CJSJ,
                    DT_ZHXGSJ
                ) VALUES (%s, %s, %s, %s, %s, %s) AS incoming
                ON DUPLICATE KEY UPDATE
                    C_BFYR_XX = incoming.C_BFYR_XX,
                    C_FYR_XX = incoming.C_FYR_XX,
                    C_WTSD_QC = incoming.C_WTSD_QC,
                    DT_CJSJ = incoming.DT_CJSJ,
                    DT_ZHXGSJ = incoming.DT_ZHXGSJ
                """.format(xfj_table=settings.mysql_xfj_table),
                xfj_rows,
            )
            await cursor.executemany(
                """
                INSERT INTO {wtxx_table} (
                    C_BH,
                    C_XFJ_BH,
                    LC_YJMS,
                    DT_CJSJ,
                    DT_ZHXGSJ
                ) VALUES (%s, %s, %s, %s, %s) AS incoming
                ON DUPLICATE KEY UPDATE
                    C_XFJ_BH = incoming.C_XFJ_BH,
                    LC_YJMS = incoming.LC_YJMS,
                    DT_CJSJ = incoming.DT_CJSJ,
                    DT_ZHXGSJ = incoming.DT_ZHXGSJ
                """.format(wtxx_table=settings.mysql_wtxx_table),
                wtxx_rows,
            )
        print("Inserted or updated {0} sample cases.".format(len(legacy_rows)))
    finally:
        connection.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Seed fixed MySQL test cases.")
    parser.add_argument("--create-schema", action="store_true", help="Create test table first.")
    parser.add_argument("--truncate", action="store_true", help="Truncate table before seeding.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(seed_cases(parse_args()))
