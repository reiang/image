import numpy as np


def formulate_state(bpp, loss, cover_score, message_density, UIQI=0, RS=True):
    state = [bpp, loss, cover_score, message_density, RS, UIQI]
    return state


def compute_reward(psnr, ssim, consumption, ):
    # PSNR 均值是 SSIM 的 10 倍
    w1 = 0.2
    w2 = 2
    w3 = 0.1
    reward = w1 * psnr + w2 * ssim - w3 * consumption
    return reward

def h0_formulate_state(h1_action,bpp, loss, cover_score, message_density,UIQI=0, RS=True):
    state = [bpp, loss, cover_score, message_density, RS, UIQI,h1_action]
    return state

#消息密度 这里是消息中1的比重 感觉
#cover_score 感觉没必要