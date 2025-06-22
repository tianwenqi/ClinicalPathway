import pandas as pd
import pm4py


# 1. 数据准备
dataframe = pd.read_csv("/home/vipuser/桌面/js/diagnosis.csv",
                        names=['case:concept:name','time:timestamp','concept:name','period'],
                        skiprows=1)
dataframe['time:timestamp'] = pd.to_datetime(dataframe['time:timestamp'])

# 按照PM4Py要求的格式准备事件日志
dataframe = pm4py.format_dataframe(dataframe, 
                                   case_id='case:concept:name', 
                                   activity_key='concept:name', 
                                   timestamp_key='time:timestamp')
event_log = pm4py.convert_to_event_log(dataframe)


'''
net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)

pm4py.view_petri_net(net, initial_marking, final_marking )

from pm4py.objects.petri_net.utils import petri_utils

# 获取所有 transition（转换），包括人工任务、系统任务等
all_transitions = net.transitions

# 筛选出有标签（即不是 silent/τ）的 transitions，通常才是业务相关的“项目”
labeled_transitions = [t for t in all_transitions if t.label is not None]

# 输出所有项目名（即 transition 的 label）
project_names = [t.label for t in labeled_transitions]
print(project_names)

'''
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

# 第一步：统计每个活动的频次
activity_counts = attributes_get.get_attribute_values(event_log, "concept:name")

# 设置阈值，比如只保留出现次数 > 100 的活动
threshold = 600
high_freq_activities = [act for act, count in activity_counts.items() if count > threshold]
high_freq_activities = [
'X线检查','X线计算机体层(CT)扫描','一般手术器材','一般营养药','免疫功能测定','其它神经系统用药','内分泌系统手术','冰冻切片与快速石蜡切片检查与诊断',
'凝血检查','口腔病药物','可吸收性止血防粘连材料','固定材料','安定药','尿液一般检查','床位费','引流器材','彩色多普勒超声检查','心电生理和心功能检查',
'心肌疾病的实验诊断','感染免疫学检测','抗出血药','护理费','换药','无机元素测定','有源手术设备配件','气管导管','气管插管辅助器材','氧气吸入',
'治疗与胃酸分泌相关疾病的药物','治疗功能性胃肠疾病的药物','注射','激素测定','特殊染色诊断技术','甲状腺治疗药','矿物质补充剂','社区卫生服务',
'糖及其代谢物测定','糖尿病用药','组织病理学检查与诊断','维生素类','肌肉弛缓药','肝病的实验诊断','肾脏疾病的实验诊断','肿瘤相关抗原测定','胃肠减压',
'自身免疫病的实验诊断','蛋白质测定','血液一般检查','血液代用品和灌注液','诊查费','输液器材','镇痛药','麻醉','麻醉、呼吸机相关器材','麻醉药'
    ]
# 第二步：过滤 event log，只保留高频活动
filtered_log = attributes_filter.apply_events(event_log, high_freq_activities,parameters={
    "attribute_key": "concept:name"})

# 第三步：使用过滤后的日志发现 Petri 网
# net, im, fm = inductive_miner.apply(filtered_log)
net, im, fm = pm4py.discover_petri_net_inductive(filtered_log)
# （可选）查看剩余项目
project_names = [t.label for t in net.transitions if t.label is not None]
print(project_names)

pm4py.view_petri_net(net, im, fm )

'''
from pm4py.statistics.attributes.log import get as attributes_get

activity_counts = attributes_get.get_attribute_values(event_log, "concept:name")
# 排序并取前 N 个活动
top_n = 20
top_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
top_activities_names = [x[0] for x in top_activities]

# 过滤日志
filtered_log = attributes_filter.apply_events(event_log, parameters={
    "attribute_key": "concept:name",
    "values": top_activities_names
})

# 再生成 Petri 网
net, im, fm = inductive_miner.apply(filtered_log)

'''




