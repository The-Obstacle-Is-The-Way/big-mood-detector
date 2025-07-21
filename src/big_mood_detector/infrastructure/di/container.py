"""
Dependency Injection Container

Implements IoC container for dependency management.
Following SOLID principles, especially Dependency Inversion.
"""

import inspect
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import (
    Any,
    cast,
    get_args,
    get_origin,
)

from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)

class CircularDependencyError(Exception):
    """Raised when circular dependency is detected."""

    pass


class DependencyNotFoundError(Exception):
    """Raised when dependency cannot be resolved."""

    pass


class Lazy[T]:
    """Lazy wrapper for dependencies."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._value: T | None = None
        self._initialized = False
        self._lock = threading.Lock()

    @property
    def value(self) -> T:
        """Get the lazy-initialized value."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._value = self._factory()
                    self._initialized = True

        # At this point, _value is guaranteed to be initialized
        assert self._value is not None, "Lazy value not properly initialized"
        return self._value


class Provide:
    """Marker for dependency injection in function parameters."""

    def __class_getitem__[T](cls, item: type[T]) -> type[T]:
        """Support Provide[ServiceType] syntax."""
        return item


class Lifetime:
    """Enum for dependency lifetimes."""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Describes a registered service."""

    def __init__(
        self,
        service_type: type,
        implementation: type | Callable | Any,
        lifetime: str,
        name: str | None = None,
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.name = name
        self.instance: Any | None = None


class Scope:
    """Represents a dependency injection scope."""

    def __init__(self, container: "Container"):
        self.container = container
        self.instances: dict[str, Any] = {}
        self._resolving: set[str] = set()  # Use string keys

    def resolve[T](self, service_type: type[T], name: str | None = None) -> T:
        """Resolve a dependency within this scope."""
        key = self._get_key(service_type, name)

        # Check for circular dependency using string key
        if key in self._resolving:
            raise CircularDependencyError(
                f"Circular dependency detected for {service_type.__name__}"
            )

        # Check if already resolved in this scope
        if key in self.instances:
            return cast(T, self.instances[key])

        # Get descriptor from container
        descriptor = self.container._get_descriptor(service_type, name)
        if not descriptor:
            raise DependencyNotFoundError(f"No registration found for {service_type}")

        # Handle scoped lifetime
        if descriptor.lifetime == Lifetime.SCOPED:
            self._resolving.add(key)
            try:
                instance = self._create_instance(descriptor)
                self.instances[key] = instance
                return cast(T, instance)
            finally:
                self._resolving.discard(key)

        # Delegate to container for other lifetimes
        return self.container.resolve(service_type, name)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance of the service."""
        return self.container._create_instance(descriptor, self)

    def _get_key(self, service_type: type, name: str | None) -> str:
        """Get cache key for service."""
        # Handle Union types (e.g., Optional[X] which is Union[X, None])
        if hasattr(service_type, "__module__") and hasattr(service_type, "__name__"):
            return (
                f"{service_type.__module__}.{service_type.__name__}:{name or 'default'}"
            )
        else:
            # For Union types and other special forms, use string representation
            return f"{str(service_type)}:{name or 'default'}"


class Container:
    """
    Inversion of Control container.

    Manages dependency registration and resolution with
    support for different lifetimes and scopes.
    """

    def __init__(self) -> None:
        self._services: dict[str, ServiceDescriptor] = {}
        self._overrides: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._resolving: set[str] = set()  # Use string keys instead of types
        logger.info("container_initialized")

    def register_singleton[T](
        self,
        service_type: type[T],
        implementation: type[T] | T | None = None,
        name: str | None = None,
    ) -> None:
        """Register a singleton service."""
        self._register(
            service_type, implementation or service_type, Lifetime.SINGLETON, name
        )
        logger.debug(
            "singleton_registered", service_type=service_type.__name__, name=name
        )

    def register_transient[T](
        self,
        service_type: type[T],
        implementation: type[T] | None = None,
        name: str | None = None,
    ) -> None:
        """Register a transient service."""
        self._register(
            service_type, implementation or service_type, Lifetime.TRANSIENT, name
        )
        logger.debug(
            "transient_registered", service_type=service_type.__name__, name=name
        )

    def register_scoped[T](
        self,
        service_type: type[T],
        implementation: type[T] | None = None,
        name: str | None = None,
    ) -> None:
        """Register a scoped service."""
        self._register(
            service_type, implementation or service_type, Lifetime.SCOPED, name
        )
        logger.debug("scoped_registered", service_type=service_type.__name__, name=name)

    def register_factory[T](
        self,
        service_type: type[T],
        factory: Callable[[], T],
        lifetime: str = Lifetime.SINGLETON,
        name: str | None = None,
    ) -> None:
        """Register a service with a factory function."""
        self._register(service_type, factory, lifetime, name)
        logger.debug(
            "factory_registered",
            service_type=service_type.__name__,
            lifetime=lifetime,
            name=name,
        )

    def resolve[T](self, service_type: type[T], name: str | None = None) -> T:
        """Resolve a dependency."""
        key = self._get_key(service_type, name)

        # Check overrides first
        if key in self._overrides:
            return cast(T, self._overrides[key])

        # Check for Lazy type
        if get_origin(service_type) is Lazy:
            inner_type = get_args(service_type)[0]
            return cast(T, Lazy(lambda: self.resolve(inner_type, name)))

        # Check for circular dependency using string key
        if key in self._resolving:
            raise CircularDependencyError(
                f"Circular dependency detected for {service_type.__name__}"
            )

        descriptor = self._get_descriptor(service_type, name)
        if not descriptor:
            raise DependencyNotFoundError(f"No registration found for {service_type}")

        # Handle based on lifetime
        if descriptor.lifetime == Lifetime.SINGLETON:
            # Check if already created (with lock)
            with self._lock:
                if descriptor.instance is not None:
                    return cast(T, descriptor.instance)

            # Create instance (without holding lock to avoid deadlock)
            self._resolving.add(key)
            try:
                instance = self._create_instance(descriptor)

                # Store instance (with lock)
                with self._lock:
                    # Double-check in case another thread created it
                    if descriptor.instance is None:
                        descriptor.instance = instance
                    return cast(T, descriptor.instance)
            finally:
                self._resolving.discard(key)

        elif descriptor.lifetime == Lifetime.TRANSIENT:
            self._resolving.add(key)
            try:
                return cast(T, self._create_instance(descriptor))
            finally:
                self._resolving.discard(key)

        elif descriptor.lifetime == Lifetime.SCOPED:
            raise RuntimeError(
                "Scoped services must be resolved within a scope. "
                "Use create_scope() context manager."
            )

        # This should never happen, but satisfies type checker
        raise RuntimeError(f"Unknown lifetime: {descriptor.lifetime}")

    def override[T](
        self, service_type: type[T], instance: T, name: str | None = None
    ) -> None:
        """Override a service for testing."""
        key = self._get_key(service_type, name)
        self._overrides[key] = instance
        logger.debug(
            "service_overridden", service_type=service_type.__name__, name=name
        )

    def reset_overrides(self) -> None:
        """Reset all overrides."""
        self._overrides.clear()
        logger.debug("overrides_reset")

    @contextmanager
    def create_scope(self) -> Iterator[Scope]:
        """Create a new dependency injection scope."""
        scope = Scope(self)
        logger.debug("scope_created")
        try:
            yield scope
        finally:
            logger.debug("scope_disposed")

    def _register(
        self,
        service_type: type,
        implementation: Any,
        lifetime: str,
        name: str | None,
    ) -> None:
        """Internal registration method."""
        key = self._get_key(service_type, name)
        descriptor = ServiceDescriptor(service_type, implementation, lifetime, name)
        self._services[key] = descriptor

    def _get_descriptor(
        self, service_type: type, name: str | None
    ) -> ServiceDescriptor | None:
        """Get service descriptor."""
        key = self._get_key(service_type, name)
        descriptor = self._services.get(key)
        return descriptor

    def _get_key(self, service_type: type, name: str | None) -> str:
        """Get cache key for service."""
        # Handle Union types (e.g., Optional[X] which is Union[X, None])
        if hasattr(service_type, "__module__") and hasattr(service_type, "__name__"):
            return (
                f"{service_type.__module__}.{service_type.__name__}:{name or 'default'}"
            )
        else:
            # For Union types and other special forms, use string representation
            return f"{str(service_type)}:{name or 'default'}"

    def _create_instance(
        self, descriptor: ServiceDescriptor, scope: Scope | None = None
    ) -> Any:
        """Create an instance of the service."""
        implementation = descriptor.implementation

        # If it's already an instance, return it
        if not inspect.isclass(implementation) and not callable(implementation):
            return implementation

        # If it's a factory function
        if callable(implementation) and not inspect.isclass(implementation):
            return implementation()

        # If it's a class, instantiate with dependency injection
        if inspect.isclass(implementation):
            # Get constructor parameters
            sig = inspect.signature(implementation.__init__)
            kwargs = {}

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Get type annotation
                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    continue

                # Handle string annotations (forward references)
                if isinstance(param_type, str):
                    # Try to resolve from the same module
                    module = inspect.getmodule(implementation)
                    if module and hasattr(module, param_type):
                        param_type = getattr(module, param_type)
                    else:
                        # Skip if can't resolve
                        continue

                # Resolve dependency
                try:
                    if scope:
                        kwargs[param_name] = scope.resolve(param_type)
                    else:
                        kwargs[param_name] = self.resolve(param_type)
                except DependencyNotFoundError:
                    # Use default if available
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise

            return implementation(**kwargs)

        raise ValueError(f"Cannot create instance of {implementation}")


# Global container instance
_container: Container | None = None
_lock = threading.Lock()


@lru_cache
def get_container() -> Container:
    """Get the global container instance (singleton)."""
    global _container
    if _container is None:
        with _lock:
            if _container is None:
                _container = Container()
    return _container


def inject(func: Callable) -> Callable:
    """
    Decorator for dependency injection.

    Automatically injects dependencies for parameters with type annotations
    that have None or Provide[T] as default.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(func)
        container = get_container()

        # Inject dependencies
        for param_name, param in sig.parameters.items():
            # Skip if already provided in kwargs
            if param_name in kwargs:
                continue

            # Check if parameter has type annotation and default is None or not provided
            if param.annotation != inspect.Parameter.empty:
                if param.default is None or param.default is inspect.Parameter.empty:
                    param_type = param.annotation
                    try:
                        kwargs[param_name] = container.resolve(param_type)
                    except DependencyNotFoundError:
                        # Use default if can't resolve
                        if param.default is not inspect.Parameter.empty:
                            kwargs[param_name] = param.default

        return func(*args, **kwargs)

    return wrapper


def setup_dependencies(settings: Any) -> Container:
    """
    Set up application-wide dependencies.

    This is called once at application startup to register
    all services with their appropriate lifetimes.
    """
    container = get_container()

    # Register configuration
    container.register_singleton(type(settings), settings)

    # Register domain services
    # Load clinical thresholds configuration
    from pathlib import Path

    from big_mood_detector.domain.services.activity_sequence_extractor import (
        ActivitySequenceExtractor,
    )
    from big_mood_detector.domain.services.biomarker_interpreter import (
        BiomarkerInterpreter,
    )
    from big_mood_detector.domain.services.circadian_rhythm_analyzer import (
        CircadianRhythmAnalyzer,
    )
    from big_mood_detector.domain.services.clinical_assessment_service import (
        ClinicalAssessmentService,
    )
    from big_mood_detector.domain.services.clinical_feature_extractor import (
        ClinicalFeatureExtractor,
    )
    from big_mood_detector.domain.services.clinical_interpreter import (
        ClinicalInterpreter,
    )
    from big_mood_detector.domain.services.clinical_thresholds import (
        ClinicalThresholdsConfig,
        load_clinical_thresholds,
    )
    from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator
    from big_mood_detector.domain.services.dsm5_criteria_evaluator import (
        DSM5CriteriaEvaluator,
    )
    from big_mood_detector.domain.services.early_warning_detector import (
        EarlyWarningDetector,
    )
    from big_mood_detector.domain.services.episode_interpreter import (
        EpisodeInterpreter,
    )
    from big_mood_detector.domain.services.intervention_evaluation_service import (
        InterventionEvaluationService,
    )
    from big_mood_detector.domain.services.longitudinal_assessment_service import (
        LongitudinalAssessmentService,
    )
    from big_mood_detector.domain.services.risk_level_assessor import (
        RiskLevelAssessor,
    )
    from big_mood_detector.domain.services.sleep_window_analyzer import (
        SleepWindowAnalyzer,
    )
    from big_mood_detector.domain.services.sparse_data_handler import SparseDataHandler
    from big_mood_detector.domain.services.treatment_recommender import (
        TreatmentRecommender,
    )

    config_path = Path("config/clinical_thresholds.yaml")
    if not config_path.exists():
        # Try relative to data directory
        fallback_path = (
            getattr(settings, "DATA_DIR", Path("data"))
            / "config"
            / "clinical_thresholds.yaml"
        )
        logger.debug(
            "clinical_config_path_fallback",
            primary_path=str(config_path),
            fallback_path=str(fallback_path),
        )
        config_path = fallback_path

    if config_path.exists():
        logger.info("loading_clinical_config", path=str(config_path))
        clinical_config = load_clinical_thresholds(config_path)
        container.register_singleton(ClinicalThresholdsConfig, clinical_config)
    else:
        logger.warning(
            "clinical_config_not_found",
            attempted_paths=["config/clinical_thresholds.yaml", str(config_path)],
        )

    # Register core services first (many others depend on these)
    container.register_singleton(DSM5CriteriaEvaluator)
    container.register_singleton(TreatmentRecommender)
    container.register_singleton(EpisodeInterpreter)
    container.register_singleton(BiomarkerInterpreter)
    container.register_singleton(RiskLevelAssessor)
    container.register_singleton(EarlyWarningDetector)

    # Register new services extracted from clinical_interpreter
    container.register_singleton(ClinicalAssessmentService)
    container.register_singleton(LongitudinalAssessmentService)
    container.register_singleton(InterventionEvaluationService)

    # Register clinical interpreter (depends on many services)
    container.register_singleton(ClinicalInterpreter)

    # Register other domain services
    container.register_singleton(SleepWindowAnalyzer)
    container.register_singleton(ActivitySequenceExtractor)
    container.register_singleton(CircadianRhythmAnalyzer)
    container.register_singleton(DLMOCalculator)
    container.register_singleton(SparseDataHandler)
    container.register_singleton(ClinicalFeatureExtractor)

    # Register feature engineering orchestrator
    from big_mood_detector.domain.services.feature_engineering_orchestrator import (
        FeatureEngineeringOrchestrator,
    )

    container.register_singleton(FeatureEngineeringOrchestrator)

    # Register application services
    from big_mood_detector.application.services.aggregation_pipeline import (
        AggregationPipeline,
    )
    from big_mood_detector.application.services.data_parsing_service import (
        DataParsingService,
    )
    from big_mood_detector.application.use_cases.process_health_data_use_case import (
        MoodPredictionPipeline,
    )

    container.register_singleton(DataParsingService)
    container.register_singleton(AggregationPipeline)
    container.register_singleton(MoodPredictionPipeline)

    # Register infrastructure services
    from big_mood_detector.infrastructure.background.task_queue import TaskQueue

    # TaskWorker will be added when implemented

    container.register_singleton(TaskQueue)

    # Register repositories
    from pathlib import Path

    from big_mood_detector.domain.repositories.activity_repository import (
        ActivityRepositoryInterface,
    )
    from big_mood_detector.domain.repositories.baseline_repository_interface import (
        BaselineRepositoryInterface,
    )
    from big_mood_detector.domain.repositories.heart_rate_repository import (
        HeartRateRepositoryInterface,
    )
    from big_mood_detector.domain.repositories.sleep_repository import (
        SleepRepositoryInterface,
    )
    from big_mood_detector.infrastructure.repositories.file_activity_repository import (
        FileActivityRepository,
    )
    from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
        FileBaselineRepository,
    )
    from big_mood_detector.infrastructure.repositories.file_heart_rate_repository import (
        FileHeartRateRepository,
    )
    from big_mood_detector.infrastructure.repositories.file_sleep_repository import (
        FileSleepRepository,
    )

    # Repository data directory from settings
    data_dir = getattr(settings, "data_dir", Path("data"))
    baselines_dir = data_dir / "baselines"

    # Register concrete implementations
    container.register_singleton(
        FileActivityRepository, lambda: FileActivityRepository(data_dir)
    )
    container.register_singleton(
        FileHeartRateRepository, lambda: FileHeartRateRepository(data_dir)
    )
    container.register_singleton(
        FileSleepRepository, lambda: FileSleepRepository(data_dir)
    )
    container.register_singleton(
        FileBaselineRepository, lambda: FileBaselineRepository(baselines_dir)
    )

    # Register interfaces to their implementations
    container.register_singleton(
        ActivityRepositoryInterface, lambda: container.resolve(FileActivityRepository)
    )
    container.register_singleton(
        HeartRateRepositoryInterface, lambda: container.resolve(FileHeartRateRepository)
    )
    container.register_singleton(
        SleepRepositoryInterface, lambda: container.resolve(FileSleepRepository)
    )
    container.register_singleton(
        BaselineRepositoryInterface, lambda: container.resolve(FileBaselineRepository)
    )

    logger.info("dependencies_configured", service_count=len(container._services))

    return container
