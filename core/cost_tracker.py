import logging


class CostTrackerMixin:

    def _track_cost_usd(self, response) -> None:
        _log = logging.getLogger(__name__)
        try:
            usage = getattr(response, "usage", None)
            if not usage:
                _log.debug(
                    "[%s] cost tracking: response has no usage object, skipping",
                    self.config.name,
                )
                return
            hidden = getattr(response, "_hidden_params", None)
            cost = (hidden.get("response_cost", 0)
                    if isinstance(hidden, dict) else 0)
            if cost:
                self._total_cost_usd += cost
            else:
                try:
                    from litellm import completion_cost
                    cost = completion_cost(completion_response=response)
                    self._total_cost_usd += cost
                except Exception as e:
                    from core.utils import estimate_llm_cost
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    estimated = estimate_llm_cost(
                        self.config.model, input_tokens, output_tokens)
                    self._total_cost_usd += estimated
                    _log.warning(
                        "[%s] cost tracking: litellm.completion_cost() failed (%s), "
                        "fell back to model-aware pricing estimate: $%.6f",
                        self.config.name, e, estimated,
                    )
        except Exception as e:
            _log.warning(
                "[%s] cost tracking: unexpected error, this LLM call recorded as $0: %s",
                self.config.name, e,
            )