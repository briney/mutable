# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for FlowNetwork and MutableFlowMatching."""

import torch

from mutable.config import MutableConfig, FlowMatchingConfig
from mutable.models.flow_matching import FlowNetwork, MutableFlowMatching
from mutable.outputs import FlowMatchingOutput


class TestFlowNetwork:
    @staticmethod
    def _make_flow_network():
        config = FlowMatchingConfig(
            hidden_size=64, num_layers=2, num_attention_heads=4,
            intermediate_size=128, time_embedding_dim=32, mu_embedding_dim=32,
            dropout=0.0,
        )
        return FlowNetwork(config)

    def test_output_shape(self):
        net = self._make_flow_network()
        noisy = torch.randn(2, 8, 64)
        germline = torch.randn(2, 8, 64)
        t = torch.rand(2)
        mu = torch.rand(2)
        out = net(noisy, germline, t, mu)
        assert out.shape == (2, 8, 64)

    def test_different_t_mu_produce_different_output(self):
        net = self._make_flow_network()
        # Initialize non-trivially
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p)
        noisy = torch.randn(2, 8, 64)
        germline = torch.randn(2, 8, 64)
        t1 = torch.tensor([0.1, 0.1])
        t2 = torch.tensor([0.9, 0.9])
        mu = torch.rand(2)
        out1 = net(noisy, germline, t1, mu)
        out2 = net(noisy, germline, t2, mu)
        assert not torch.allclose(out1, out2)

    def test_time_embedding_dimensions(self):
        config = FlowMatchingConfig(
            hidden_size=64, num_layers=1, num_attention_heads=4,
            time_embedding_dim=48, mu_embedding_dim=16,
            dropout=0.0,
        )
        net = FlowNetwork(config)
        # conditioning_dim = 48 + 16 = 64
        assert net.conditioning_proj.in_features == 64


class TestMutableFlowMatching:
    def test_forward_returns_loss(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2
        germ_mask = torch.ones(B, L, dtype=torch.long)
        mut_mask = torch.ones(B, L, dtype=torch.long)
        mu = torch.rand(B)

        small_flow_model.train()
        out = small_flow_model(
            germline_input_ids=germ_ids, germline_attention_mask=germ_mask,
            mutated_input_ids=mut_ids, mutated_attention_mask=mut_mask,
            mu=mu, return_dict=True,
        )
        small_flow_model.eval()
        assert isinstance(out, FlowMatchingOutput)
        assert out.loss is not None
        assert out.loss.dim() == 0

    def test_velocity_shapes(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2

        small_flow_model.train()
        out = small_flow_model(
            germline_input_ids=germ_ids,
            mutated_input_ids=mut_ids,
            return_dict=True,
        )
        small_flow_model.eval()
        assert out.predicted_velocity.shape == (B, 8, 64)
        assert out.target_velocity.shape == (B, 8, 64)

    def test_default_mu_uses_zeros(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2

        small_flow_model.train()
        out = small_flow_model(
            germline_input_ids=germ_ids,
            mutated_input_ids=mut_ids,
            mu=None,
            return_dict=True,
        )
        small_flow_model.eval()
        assert out.loss is not None

    def test_generate_produces_token_ids(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        germ_mask = torch.ones(B, L, dtype=torch.long)

        gen = small_flow_model.generate(
            germline_input_ids=germ_ids,
            germline_attention_mask=germ_mask,
            num_steps=2,
            solver="euler",
        )
        assert gen.dtype == torch.long
        assert gen.shape == (B, L)

    def test_generate_with_custom_solver(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2

        gen = small_flow_model.generate(
            germline_input_ids=germ_ids,
            num_steps=2,
            solver="rk4",
        )
        assert gen.dtype == torch.long

    def test_backbone_frozen_in_forward(self, small_flow_model):
        # The backbone encoding happens inside torch.no_grad()
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2

        small_flow_model.train()
        out = small_flow_model(
            germline_input_ids=germ_ids,
            mutated_input_ids=mut_ids,
            return_dict=True,
        )
        out.loss.backward()
        small_flow_model.eval()

        # Backbone encoder params should have no gradient
        for p in small_flow_model.mutable.encoder.parameters():
            assert p.grad is None

    def test_flow_network_has_gradient(self, small_config, small_flow_config):
        model = MutableFlowMatching(small_config, small_flow_config)
        model.train()
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2

        out = model(
            germline_input_ids=germ_ids,
            mutated_input_ids=mut_ids,
            return_dict=True,
        )
        out.loss.backward()

        # Flow network params should have gradients
        has_grad = False
        for p in model.flow_network.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad
