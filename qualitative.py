import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
#这里导入你自己的数据

#......
x_axix = [0.9, 1, 1.1, 1.2]

# # CIDEr
# y_axix = [51.4, 52.0, 52.6, 51.1]
# y_axix_1 = [115.1, 119.1, 124.4, 115.8]
#
# #B4
# y_b4 = [41.9, 43.3, 42.6, 41]
# y_b4_1 = [52.6, 55.9, 59, 54.3]
# #......
#
# #x_axix，train_pn_dis这些都是长度相同的list()
#
# #开始画图
#
# # sub_axix = filter(lambda x:x%200 == 0, x_axix)
#
# # plt.title('Result Analysis')
#
# plt.plot(x_axix, y_b4_1, linestyle='-', marker='o', markerfacecolor='dodgerblue', markersize='10', color='khaki', linewidth=5)
#
# front = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18}
#
# plt.xlabel('λ', front)
#
# plt.ylabel('BLUE@4', front)
# # plt.ylabel('CIDEr', front)
#
# # x_major_locator=MultipleLocator(0.1)
# # y_major_locator=MultipleLocator(2)
# ax=plt.gca()
# #ax为两条坐标轴的实例
# # ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
# plt.figure(figsize=(6, 4))
#
# plt.xticks(fontproperties='Times New Roman', size=15)
# plt.yticks(fontproperties='Times New Roman', size=15)
#
# # plt.savefig('/Users/zhangyi/Desktop/MSVD_B.pdf', bbox_inches='tight')
# plt.show()
