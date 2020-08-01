from matplotlib import pyplot as plt
from collections import defaultdict

from data_gen import infest_to_class_ind

def gen_num_image_per_class(image_infest_list, n_classes):
	class_ind_to_image_num = defaultdict(int)
	for _, infest in image_infest_list:
		class_ind_to_image_num[infest_to_class_ind(n_classes, infest)] += 1

	return sorted(list(class_ind_to_image_num.items()), key=lambda t: t[1])

def plot_ROC(fpr_keras, tpr_keras, auc_keras):
	plt.style.use("ggplot")
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig('ROC_pos_vs_neg')
	# plt.show()

def plot_histogram(image_infest_list):
	_, infests = zip(*image_infest_list)
	hist, bin_edges = np.histogram(infests, bins=15)
	plt.bar(bin_edges[:-1], hist, width = 0.05, color='#0504aa')
	plt.xlim(min(bin_edges), max(bin_edges))
	plt.grid(axis='y')
	plt.xlabel('Infestation level',fontsize=8)
	plt.ylabel('#images',fontsize=8)
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)
	plt.title('Dataset Distribution',fontsize=10)
	plt.show()

def plot_histogram_multiclasses(image_infest_list):
	bin_edges, hist = zip(*gen_num_image_per_class(image_infest_list))
	plt.bar(bin_edges, hist, width = 0.5, color='#0504aa')
	plt.xlim(-1, n_classes + 0.5)
	plt.grid(axis='y')
	plt.xlabel('Value',fontsize=10)
	plt.ylabel('#images',fontsize=10)
	if n_classes > 2:
		step = 1.0 / (n_classes - 1)
		step = float("{0:0.1f}".format(step))
		end_step = step * (n_classes - 1)
		l = '0'
		x_labels = []
		for h in np.arange(0, end_step, step):
			h = "{0:0.1f}".format(h)
			x_labels.append(l + ',' + h)
			l = h
		x_labels.append(l + ',' + '1')
		plt.xticks(np.arange(0, n_classes + 1), tuple(x_labels), fontsize=8)
	else:
		plt.xticks(fontsize=8)
	plt.yticks(fontsize=10)
	plt.title('Multiple Classes Histogram',fontsize=15)
	plt.show()