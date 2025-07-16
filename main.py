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
    with open(proto_file, 'r', encoding='utf-8') as file:
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

# 内存估算函数
def estimate_memory_usage(n1, n2, dtype=np.float64):
    """估算节点匹配所需内存"""
    items = n1 * n2  # 相似度矩阵元素数量
    bytes_per_item = dtype().itemsize
    return items * bytes_per_item

def get_available_memory():
    """获取系统可用内存（保留20%作为安全裕度）"""
    total_mem = psutil.virtual_memory().total
    available_mem = psutil.virtual_memory().available
    safe_mem = available_mem * 0.8  # 保留20%的安全裕度
    return safe_mem

def calculate_chunk_size(n1, n2, dtype=np.float64):
    """计算分块大小"""
    item_size = dtype().itemsize
    available_mem = get_available_memory()
    
    # 计算单个分块的最大元素数
    max_items_per_chunk = int(available_mem // item_size)
    
    # 计算分块数量（行方向）
    chunks = max(1, (n1 * n2) // max_items_per_chunk + 1)
    chunk_size = max(1, n1 // chunks)
    
    logger.info(f"内存统计: 可用内存={available_mem/1e9:.2f}GB, 安全内存={available_mem/1e9:.2f}GB")
    logger.info(f"矩阵大小: {n1}x{n2}={n1*n2} 元素, 所需内存={estimate_memory_usage(n1, n2)/1e9:.2f}GB")
    logger.info(f"将拆分为 {chunks} 个块, 每块约 {chunk_size} 行")
    
    return chunk_size

async def perform_graph_matching(G1, G2, G1_n, G2_n, args):
    n1 = len(G1_n)
    n2 = len(G2_n)
    
    # 计算分块大小
    chunk_size = calculate_chunk_size(n1, n2)
    logger.info(f"节点匹配分块大小: {chunk_size}行/块 (共{n1}行)")
    
    # 初始化相似度矩阵
    S = np.zeros((n1, n2))
    total_chunks = (n1 + chunk_size - 1) // chunk_size
    
    # 分块计算节点相似度
    for chunk_idx in range(total_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min((chunk_idx + 1) * chunk_size, n1)
        rows_in_chunk = end_row - start_row
        
        logger.info(f"处理节点块 {chunk_idx+1}/{total_chunks} (行 {start_row}-{end_row-1})")
        
        # 分块计算
        start_time = time.time()
        with tqdm(total=rows_in_chunk, desc=f"节点块 {chunk_idx+1}/{total_chunks}", 
                 bar_format="{l_bar}{bar}|{postfix}") as pbar:
            for i_local in range(rows_in_chunk):
                i = start_row + i_local
                if args.insert_debug_delay:
                    await asyncio.sleep(0.1)
                
                # 内存监控
                current_mem = psutil.virtual_memory()
                if current_mem.percent > 95:
                    logger.warning(f"内存使用过高! {current_mem.percent}%, 暂停处理")
                    await asyncio.sleep(1)
                
                for j in range(n2):
                    # 名称完全相同的节点强制匹配
                    if G1_n[i] == G2_n[j]:
                        S[i, j] = 1.0
                    else:
                        S[i, j] = 0.0  # 其他情况初始化为0
                
                # 更新进度
                if i_local % 10 == 0 or i_local == rows_in_chunk - 1:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rows_per_sec = (i_local + 1) / elapsed if elapsed > 0 else 0
                    items_per_sec = rows_per_sec * G2.number_of_nodes()
                    ms_per_item = 1000 / items_per_sec if items_per_sec > 0 else 0
                    cpu_percent_current = psutil.cpu_percent()
                    pbar.set_postfix_str(f"第{i+1}个，共{len(G1_n)}个，速度{items_per_sec:.2f}项每秒，即{ms_per_item:.0f}毫秒每项，CPU利用率{cpu_percent_current:.0f}%，内存占用: {current_mem.percent}%")
                pbar.update(1)
    
    # 匈牙利算法全局匹配
    logger.info("使用匈牙利算法进行全局匹配...")
    row_ind, col_ind = linear_sum_assignment(-S)
    
    node_matches = []
    for i, j in zip(row_ind, col_ind):
        node1 = G1_n[i]
        node2 = G2_n[j]
        sim_score = S[i, j]
        node_matches.append((node1, node2, sim_score))
    
    return S, node_matches

async def perform_edge_matching(G1, G2, G1_n, G2_n, S, args):
    # 将此函数修改为异步生成器
    # yield from edge_matches

    logger.info("开始计算字段名的字符串相似度...")
    def name_similarity(name1, name2):
        return SequenceMatcher(None, name1, name2).ratio()

    logger.info("开始构建边列表...")
    if args.insert_debug_delay:
        await asyncio.sleep(0.1)
    E1 = list(G1.edges(data=True))
    E2 = list(G2.edges(data=True))
    
    # 计算分块大小
    chunk_size = calculate_chunk_size(len(E1), len(E2))
    logger.info(f"边匹配分块大小: {chunk_size}行/块 (共{len(E1)}行)")
    total_chunks = (len(E1) + chunk_size - 1) // chunk_size
    
    # 初始化边相似度矩阵

    
    # 分块处理边
    for chunk_idx in range(total_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min((chunk_idx + 1) * chunk_size, len(E1))
        rows_in_chunk = end_row - start_row
        
        logger.info(f"处理边块 {chunk_idx+1}/{total_chunks} (边 {start_row}-{end_row-1})")
        
        # 为当前块创建相似度矩阵
        S_block = np.zeros((rows_in_chunk, len(E2)))
        
        start_time = time.time()
        with tqdm(total=rows_in_chunk, desc=f"边块 {chunk_idx+1}/{total_chunks}", 
                 bar_format="{l_bar}{bar}|{postfix}") as pbar:
            for i_local in range(rows_in_chunk):
                i = start_row + i_local
                if args.insert_debug_delay:
                    await asyncio.sleep(0.1)
                
                # 内存监控
                current_mem = psutil.virtual_memory()
                if current_mem.percent > 95:
                    logger.warning(f"内存使用过高! {current_mem.percent}%, 暂停处理")
                    await asyncio.sleep(1)
                
                u1, v1, attr1 = E1[i]
                for j, (u2, v2, attr2) in enumerate(E2):
                    idx_u1, idx_v1 = G1_n.index(u1), G1_n.index(v1)
                    idx_u2, idx_v2 = G2_n.index(u2), G2_n.index(v2)
                    
                    # 计算顶点匹配相似度
                    sim_uv = 0.5 * (
                        (S[idx_u1, idx_u2] + S[idx_v1, idx_v2]) +
                        (S[idx_u1, idx_v2] + S[idx_v1, idx_u2])
                    )
                    
                    # 计算字段名相似度
                    name1 = attr1.get('attr', '')
                    name2 = attr2.get('attr', '')
                    name_sim = name_similarity(name1, name2) if name1 and name2 else 0.0
                    
                    # 融合权重
                    S_block[i_local, j] = 0.7 * sim_uv + 0.3 * name_sim
                
                # 更新进度
                if i_local % 10 == 0 or i_local == rows_in_chunk - 1:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rows_per_sec = (i_local + 1) / elapsed if elapsed > 0 else 0
                    items_per_sec = rows_per_sec * G2.number_of_nodes()
                    ms_per_item = 1000 / items_per_sec if items_per_sec > 0 else 0
                    cpu_percent_current = psutil.cpu_percent()
                    pbar.set_postfix_str(f"第{i+1}个，共{len(G1_n)}个，速度{items_per_sec:.2f}项每秒，即{ms_per_item:.0f}毫秒每项，CPU利用率{cpu_percent_current:.0f}%，内存占用: {current_mem.percent}%")
                pbar.update(1)
        
        # 对当前块执行匈牙利算法
        logger.info(f"对边块 {chunk_idx+1}/{total_chunks} 执行匈牙利算法...")
        row_ind, col_ind = linear_sum_assignment(-S_block)
        
        # 处理当前块的匹配结果
        for i_local, j in zip(row_ind, col_ind):
            i_global = start_row + i_local
            (u1, v1, attr1) = E1[i_global]
            (u2, v2, attr2) = E2[j]
            score = S_block[i_local, j]
            
            if attr1.get('attr', '') == '':
                continue
            
            yield (u1, v1, attr1, u2, v2, attr2, score)
    


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

    # 在节点匹配前添加内存检查
    required_mem = estimate_memory_usage(len(G1_n), len(G2_n))
    available_mem = get_available_memory()
    
    logger.info(f"内存需求: {required_mem/1e9:.2f}GB, 可用内存: {available_mem/1e9:.2f}GB")
    
    if required_mem > available_mem:
        logger.warning("所需内存超过可用内存，将使用分块处理...")
    
    S, node_matches = await perform_graph_matching(G1, G2, G1_n, G2_n, args)
    logger.info("---节点匹配结果---")
    for node1, node2, sim_score in node_matches:
        if args.hide_complete_match and sim_score >= 0.99:
            continue
        logger.info(f"proto1:\"{node1}\"<-相似度:{sim_score:.2%}->proto2:\"{node2}\"")
    logger.info("-" * 12) # 添加分割线

    edge_matches_generator = perform_edge_matching(G1, G2, G1_n, G2_n, S, args)
    logger.info("---边匹配结果---")
    async for u1, v1, attr1, u2, v2, attr2, score in edge_matches_generator:
        if args.hide_complete_match and score >= 0.99:
            continue
        logger.info(f'proto1:"{u1}"-[{attr1.get("attr", "")}]-"{v1}"<-相似度:{score:.2%}->proto2:"{u2}"-[{attr2.get("attr", "")}]-"{v2}"')
    logger.info("-" * 11) # 添加分割线

if __name__ == "__main__":
    asyncio.run(main())
