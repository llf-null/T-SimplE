# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
def shredFacts(facts):
    heads      = torch.tensor(facts[:,0]).long()
    rels       = torch.tensor(facts[:,1]).long()
    tails      = torch.tensor(facts[:,2]).long()
    dates=torch.tensor(facts[:,3]).long()
    return heads,rels,tails,dates
