import sys
import numpy as np
import pygmtools as pygm
import networkx as nx
import re
import os
import asyncio
import argparse
import time
import psutil
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
from loguru import logger
from tqdm.asyncio import tqdm

pygm.set_backend('numpy')
np.random.seed(1)

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# 定义日志文件路径
LOG_FILE_PATH = "./log.txt"

# 解析 proto 文件中的 message 和 enum
def parse_messages_fx(content):
    pl_messages = []
    current_message_content = []
    in_block = False
    brace_level = 0

    lines = content.splitlines()
    for index, line in enumerate(lines):
        stripped_line = line.strip()

        if not in_block and ('message ' in stripped_line or 'enum ' in stripped_line):
            is_enum = 'enum ' in stripped_line
            in_block = True
            current_message_content = [line]
            brace_level = line.count('{') - line.count('}')
            # 尝试获取前一行的注释作为当前 message/enum 的注释
            current_comment = ''
            if index > 0 and lines[index - 1].strip().startswith('//'):
                current_comment = lines[index - 1].strip()
        elif in_block:
            current_message_content.append(line)
            brace_level += line.count('{') - line.count('}')

            if brace_level == 0:
                in_block = False
                full_message_block = '\n'.join(current_message_content)
                match = re.search(r'(message|enum)\s+(\w+)', full_message_block)
                if match:
                    name = match.group(2)
                    pl_message = {
                        'is_enum': is_enum,
                        'name': name,
                        'cmd_id': -1,
                        'layer': 0 if is_enum else -1,
                        'base_widget': 0,
                        'widget': -1,
                        'count': 0,
                        'wcount': 0,
                        'imports': [],
                        'comment': current_comment,
                        'message': full_message_block.strip()
                    }
                    pl_messages.append(pl_message)
                current_message_content = []

    return pl_messages

# 将 proto 文件解析为图结构
def parse_proto_to_graph(proto_file):
    with open(proto_file, 'r') as file:
        proto_data = file.read()

    basic_type_set = {'int32', 'int64', 'uint32', 'uint64', 'float', 'fixed32', 'fixed64', 'sfixed32', 'sfixed64', 'bool', 'string', 'bytes'}
    message_name_set = set()
    enum_name_set = set()

    lines = proto_data.splitlines()
    field_pattern = re.compile(r'\w+ (\w+) = \d+;', re.MULTILINE)
    for line in lines:
        if line.strip().startswith('message'):
            name = line.split(' ')[1].split('{')[0]
            message_name_set.add(name)
        elif line.strip().startswith('enum'):
            name = line.split(' ')[1].split('{')[0]
            enum_name_set.add(name)

    graph = nx.DiGraph()

    # 添加描述顶点
    graph.add_node('node_enum')
    graph.add_node('node_message')

    # 添加所有类型名称作为图的顶点
    for name in basic_type_set:
        graph.add_node(name)
    for name in enum_name_set:
        graph.add_node(name)
    for name in message_name_set:
        graph.add_node(name)

    # 添加描述顶点到具体类型的边
    for name in enum_name_set:
        graph.add_edge('node_enum', name)
    for name in message_name_set:
        graph.add_edge('node_message', name)

    # 添加 message 内部字段的边
    messages = parse_messages_fx(proto_data)
    for message in messages:
        if message['is_enum']:
            continue
        lines = message['message'].splitlines()
        for line in lines:
            matchObj = field_pattern.search(line)
            if matchObj:
                field_name = matchObj.group(1)
                line_sub = line.split(field_name)[0]
                
                # 查找字段类型并添加边
                for name_set in [basic_type_set, enum_name_set, message_name_set]:
                    for name in name_set:
                        if name in line_sub:
                            graph.add_edge(message['name'], name, attr=field_name)

    return graph

# 获取 proto 文件路径，支持命令行参数或用户输入
def get_proto_file_path(arg_index, prompt_message):
    if len(sys.argv) > arg_index:
        file_path = sys.argv[arg_index]
        if not os.path.exists(file_path):
            logger.error(f"文件 '{file_path}' 不存在。")
            sys.exit(1)
        return file_path
    else:
        while True:
            file_path = input(prompt_message)
            if os.path.exists(file_path):
                return file_path
            else:
                logger.error(f"文件 '{file_path}' 不存在，请重新输入。")

async def perform_graph_matching(G1, G2, G1_n, G2_n, args):
    logger.info("开始构建比较矩阵...")
    if args.insert_debug_delay:
        time.sleep(0.1)
    K = pygm.utils.build_aff_mat_from_networkx(G1, G2)
    logger.info("开始图匹配算法...")
    if args.insert_debug_delay:
        time.sleep(0.1)
    X = pygm.rrwm(K, n1=G1.number_of_nodes(), n2=G2.number_of_nodes(), backend='numpy')

    # 强制相同名称的节点完全匹配
    start_time = time.time()
    cpu_percent_at_start = psutil.cpu_percent(interval=None) # 获取初始CPU利用率
    with tqdm(total=len(G1_n), desc="计算节点相似度", bar_format="{l_bar}{bar}|{postfix}") as pbar:
        for i, name1 in enumerate(G1_n):
            if args.insert_debug_delay:
                time.sleep(0.1)
            current_time = time.time()
            elapsed_time = current_time - start_time
            items_per_sec = i / elapsed_time if elapsed_time > 0 else 0
            ms_per_item = (1 / items_per_sec * 1000) if items_per_sec > 0 else 0
            cpu_percent_current = psutil.cpu_percent(interval=None) # 获取当前CPU利用率
            pbar.set_postfix_str(f"第{i+1}个，共{len(G1_n)}个，速度{items_per_sec:.2f}项每秒，即{ms_per_item:.0f}毫秒每项，CPU利用率{cpu_percent_current:.0f}%")
            pbar.update(1)
        for j, name2 in enumerate(G2_n):
            if name1 == name2:
                X[i, j] = 1.0

    row_ind, col_ind = linear_sum_assignment(-X)

    node_matches = []
    for i, j in zip(row_ind, col_ind):
        node1 = G1_n[i]
        node2 = G2_n[j]
        sim_score = X[i, j]
        node_matches.append((node1, node2, sim_score))
    return X, node_matches

async def perform_edge_matching(G1, G2, G1_n, G2_n, X, args):
    logger.info("开始计算字段名的字符串相似度...")
    def name_similarity(name1, name2):
        return SequenceMatcher(None, name1, name2).ratio()

    logger.info("开始构建边列表...")
    if args.insert_debug_delay:
        time.sleep(0.1)
    E1 = list(G1.edges(data=True))
    E2 = list(G2.edges(data=True))

    logger.info("开始初始化边相似度矩阵...")
    if args.insert_debug_delay:
        time.sleep(0.1)
    S = np.zeros((len(E1), len(E2)))

    logger.info("开始构建边相似度矩阵...")
    start_time = time.time()
    cpu_percent_at_start = psutil.cpu_percent(interval=None) # 获取初始CPU利用率
    with tqdm(total=len(E1), desc="计算边相似度", bar_format="{l_bar}{bar}|{postfix}") as pbar:
        for i, (u1, v1, attr1) in enumerate(E1):
            if args.insert_debug_delay:
                time.sleep(0.1)
            current_time = time.time()
            elapsed_time = current_time - start_time
            items_per_sec = i / elapsed_time if elapsed_time > 0 else 0
            ms_per_item = (1 / items_per_sec * 1000) if items_per_sec > 0 else 0
            cpu_percent_current = psutil.cpu_percent(interval=None) # 获取当前CPU利用率
            pbar.set_postfix_str(f"第{i+1}个，共{len(E1)}个，速度{items_per_sec:.2f}项每秒，即{ms_per_item:.0f}毫秒每项，CPU利用率{cpu_percent_current:.0f}%")
            pbar.update(1)
        for j, (u2, v2, attr2) in enumerate(E2):
            idx_u1, idx_v1 = G1_n.index(u1), G1_n.index(v1)
            idx_u2, idx_v2 = G2_n.index(u2), G2_n.index(v2)

            # 计算顶点匹配相似度（考虑方向对称性）
            sim_uv = 0.5 * (
                (X[idx_u1, idx_u2] + X[idx_v1, idx_v2]) +
                (X[idx_u1, idx_v2] + X[idx_v1, idx_u2])
            )

            # 获取字段名
            name1 = attr1.get('attr', '')
            name2 = attr2.get('attr', '')

            # 计算字段名相似度
            name_sim = name_similarity(name1, name2)

            # 融合权重
            S[i, j] = 0.7 * sim_uv + 0.3 * name_sim

    edge_row_ind, edge_col_ind = linear_sum_assignment(-S)

    edge_matches = []
    for i, j in zip(edge_row_ind, edge_col_ind):
        (u1, v1, attr1) = E1[i]
        (u2, v2, attr2) = E2[j]
        score = S[i, j]
        if attr1.get('attr', '') == '': # 过滤掉没有字段名的边
            continue
        edge_matches.append((u1, v1, attr1, u2, v2, attr2, score))
    return edge_matches

async def main():
    parser = argparse.ArgumentParser(description="比较两个 proto 文件并显示相似度。")
    parser.add_argument("proto_file1", nargs='?', help="第一个 proto 文件路径")
    parser.add_argument("proto_file2", nargs='?', help="第二个 proto 文件路径")
    parser.add_argument("--hide-complete-match", action="store_true", help="隐藏完全匹配的节点和边")
    parser.add_argument("--log-to-file", action="store_true", help="将日志输出到文件")
    parser.add_argument("--console-progress-only", action="store_true", help="控制台只显示进度")
    parser.add_argument("--insert-debug-delay", action="store_true", help="在计算之间插入0.1s的延迟，用于测试")
    args = parser.parse_args()

    # 配置日志输出
    if args.log_to_file:
        logger.add(LOG_FILE_PATH, rotation="10 MB", retention="7 days", level="INFO")
        if args.console_progress_only:
            # 只显示进度相关的日志
            logger.remove()
            logger.add(sys.stderr, format="<level>{message}</level>", level="INFO", filter=lambda record: any(keyword in record["message"] for keyword in [
                "计算边相似度", "开始构建比较矩阵", "开始图匹配算法", "开始计算节点匹配结果",
                "计算节点相似度",
                "开始计算字段名的字符串相似度", "开始构建边列表", "开始初始化边相似度矩阵",
                "开始构建边相似度矩阵", "开始使用匈牙利算法匹配边"
            ]))
    elif args.console_progress_only:
        # 只显示进度相关的日志
        logger.remove()
        logger.add(sys.stderr, format="<level>{message}</level>", level="INFO", filter=lambda record: any(keyword in record["message"] for keyword in [
            "计算边相似度", "开始构建比较矩阵", "开始图匹配算法", "开始计算节点匹配结果",
            "计算节点相似度",
            "开始计算字段名的字符串相似度", "开始构建边列表", "开始初始化边相似度矩阵",
            "开始构建边相似度矩阵", "开始使用匈牙利算法匹配边"
        ]))

    proto_file1 = get_proto_file_path(1, "请输入第一个 proto 文件路径：") if args.proto_file1 is None else args.proto_file1
    proto_file2 = get_proto_file_path(2, "请输入第二个 proto 文件路径：") if args.proto_file2 is None else args.proto_file2

    G1 = parse_proto_to_graph(proto_file1)
    G2 = parse_proto_to_graph(proto_file2)

    G1_n = list(G1.nodes())
    G2_n = list(G2.nodes())

    logger.info(f"G1节点数: {len(G1_n)}")
    logger.info(f"G2节点数: {len(G2_n)}")

    confirm = input("数据量如上所示，要开始比较吗？[y/N] ")
    if confirm.lower() != 'y':
        sys.exit(0)

    X, node_matches = await perform_graph_matching(G1, G2, G1_n, G2_n, args)
    logger.info("---节点匹配结果---")
    for node1, node2, sim_score in node_matches:
        if args.hide_complete_match and sim_score >= 0.99:
            continue
        logger.info(f"proto1:\"{node1}\"<-相似度:{sim_score:.2%}->proto2:\"{node2}\"")
    logger.info("-" * 12) # 添加分割线

    edge_matches = await perform_edge_matching(G1, G2, G1_n, G2_n, X, args)
    logger.info("---边匹配结果---")
    for u1, v1, attr1, u2, v2, attr2, score in edge_matches:
        if args.hide_complete_match and score >= 0.99:
            continue
        logger.info(f'proto1:"{u1}"-[{attr1.get("attr", "")}]-"{v1}"<-相似度:{score:.2%}->proto2:"{u2}"-[{attr2.get("attr", "")}]-"{v2}"')
    logger.info("-" * 11) # 添加分割线

if __name__ == "__main__":
    asyncio.run(main())
