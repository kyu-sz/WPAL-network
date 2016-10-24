def mA(attr, gt):
	num = attr.__len__()
	num_attrs = attr[0].__len__()

	for i in xrange(num_attrs):
		print '--------------------------------------------'
		print i
		print sum([attr[j][i] for j in xrange(num)]), \
			':', sum([attr[j][i] * gt[j][i] for j in xrange(num)]), \
			':', sum([gt[j][i] for j in xrange(num)])
		print sum([attr[j][i] * gt[j][i] for j in xrange(num)]) / sum([gt[j][i] for j in xrange(num)])
		print sum([(1 - attr[j][i]) for j in xrange(num)]), \
			':', sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)]), \
			':', sum([(1 - gt[j][i]) for j in xrange(num)])
		print sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)]) / sum([(1 - gt[j][i]) for j in xrange(num)])

	mA = (sum([(
		           sum([attr[j][i] * gt[j][i] for j in xrange(num)])
		           / sum([gt[j][i] for j in xrange(num)])
		           + sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)])
		           / sum([(1 - gt[j][i]) for j in xrange(num)])
	           ) for i in xrange(num_attrs)])) / (2 * num_attrs)
	return mA
