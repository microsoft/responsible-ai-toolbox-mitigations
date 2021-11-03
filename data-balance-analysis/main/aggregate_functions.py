import numpy as np

def generalized_entropy_index(benefits, alpha, useAbsVal, errorTolerance=1e-12):
    if useAbsVal:
        benefits = np.absolute(benefits)
    benefits_mean = np.mean(benefits)
    norm_benefits = benefits / benefits_mean
    cnt = norm_benefits.size
    if abs(alpha - 1.0) < errorTolerance:
        gei = np.sum(norm_benefits * np.log(norm_benefits)) / cnt
    elif abs(alpha) < errorTolerance:
        gei = np.sum(-np.log(norm_benefits)) / cnt
    else:
        gei = np.sum(np.power(norm_benefits, alpha) - 1.0) / (cnt * alpha * (alpha - 1.0))
    return gei


def atkinson_index(benefits, epsilon, errorTolerance=1e-12):
    cnt = benefits.size
    benefits_mean = np.mean(benefits)
    norm_benefits = benefits / benefits_mean
    alpha = 1 - epsilon
    if abs(alpha) < errorTolerance:
        ati = 1.0 - np.power(np.prod(norm_benefits), 1.0 / cnt)
    else:
        power_mean = np.sum(np.power(norm_benefits, alpha)) / cnt
        ati = 1.0 - np.power(power_mean, 1.0 / alpha)
    return ati


def thiel_t_index(benefits):
    return generalized_entropy_index(benefits, 1.0, True)


def thiel_l_index(benefits):
    return generalized_entropy_index(benefits, 0.0, True)
