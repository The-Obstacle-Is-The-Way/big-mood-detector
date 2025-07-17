"""
Test Dependency Injection Container

TDD for dependency injection with proper lifecycle management.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestDependencyInjection:
    """Test dependency injection container."""

    def test_container_can_be_imported(self):
        """Test that DI container can be imported."""
        from big_mood_detector.core.dependencies import (
            Container,
            get_container,
            Provide,
            inject
        )
        
        assert Container is not None
        assert get_container is not None
        assert Provide is not None
        assert inject is not None

    def test_container_singleton_pattern(self):
        """Test that container uses singleton pattern."""
        from big_mood_detector.core.dependencies import get_container
        
        container1 = get_container()
        container2 = get_container()
        
        # Should be the same instance
        assert container1 is container2

    def test_register_and_resolve_singleton(self):
        """Test registering and resolving singleton dependencies."""
        from big_mood_detector.core.dependencies import Container
        
        # Create test service
        class TestService:
            def __init__(self):
                self.value = 42
        
        container = Container()
        
        # Register singleton
        container.register_singleton(TestService)
        
        # Resolve multiple times
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)
        
        # Should be same instance
        assert instance1 is instance2
        assert instance1.value == 42

    def test_register_and_resolve_transient(self):
        """Test registering and resolving transient dependencies."""
        from big_mood_detector.core.dependencies import Container
        
        # Create test service
        class TestService:
            def __init__(self):
                self.value = 42
        
        container = Container()
        
        # Register transient
        container.register_transient(TestService)
        
        # Resolve multiple times
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)
        
        # Should be different instances
        assert instance1 is not instance2
        assert instance1.value == 42
        assert instance2.value == 42

    def test_register_factory(self):
        """Test registering dependencies with factory functions."""
        from big_mood_detector.core.dependencies import Container
        
        # Create test service
        class TestService:
            def __init__(self, config: dict):
                self.config = config
        
        container = Container()
        
        # Register with factory
        container.register_factory(
            TestService,
            lambda: TestService({"key": "value"})
        )
        
        # Resolve
        instance = container.resolve(TestService)
        
        assert instance.config == {"key": "value"}

    def test_automatic_dependency_resolution(self):
        """Test automatic resolution of dependencies."""
        from big_mood_detector.core.dependencies import Container
        
        # Create service hierarchy
        class Repository:
            def get_data(self):
                return "data"
        
        class Service:
            def __init__(self, repo: Repository):
                self.repo = repo
            
            def process(self):
                return f"processed {self.repo.get_data()}"
        
        container = Container()
        
        # Register dependencies
        container.register_singleton(Repository)
        container.register_singleton(Service)
        
        # Resolve service (should auto-inject repo)
        service = container.resolve(Service)
        
        assert service.process() == "processed data"

    def test_inject_decorator(self):
        """Test dependency injection decorator."""
        from big_mood_detector.core.dependencies import get_container, inject, Provide
        
        # Create test service
        class TestService:
            def get_value(self):
                return 42
        
        # Use global container (what inject uses)
        container = get_container()
        container.register_singleton(TestService)
        
        # Use inject decorator - simpler approach
        @inject
        def function_with_deps(service: TestService = None):
            return service.get_value()
        
        # Call function (deps should be injected)
        result = function_with_deps()
        
        assert result == 42

    def test_override_dependencies_for_testing(self):
        """Test overriding dependencies for testing."""
        from big_mood_detector.core.dependencies import Container
        
        # Create services
        class RealService:
            def get_value(self):
                return "real"
        
        class MockService:
            def get_value(self):
                return "mock"
        
        container = Container()
        container.register_singleton(RealService)
        
        # Resolve normally
        real = container.resolve(RealService)
        assert real.get_value() == "real"
        
        # Override for testing
        container.override(RealService, MockService())
        
        # Resolve again
        mock = container.resolve(RealService)
        assert mock.get_value() == "mock"
        
        # Reset overrides
        container.reset_overrides()
        
        # Should be back to real
        real_again = container.resolve(RealService)
        assert real_again.get_value() == "real"

    def test_scoped_lifetime_management(self):
        """Test scoped lifetime management for requests."""
        from big_mood_detector.core.dependencies import Container
        
        # Create test service
        class RequestScopedService:
            def __init__(self):
                import uuid
                self.id = str(uuid.uuid4())
        
        container = Container()
        container.register_scoped(RequestScopedService)
        
        # Create scope 1
        with container.create_scope() as scope1:
            instance1a = scope1.resolve(RequestScopedService)
            instance1b = scope1.resolve(RequestScopedService)
            
            # Same instance within scope
            assert instance1a is instance1b
        
        # Create scope 2
        with container.create_scope() as scope2:
            instance2 = scope2.resolve(RequestScopedService)
            
            # Different instance in different scope
            assert instance2.id != instance1a.id

    def test_lazy_initialization(self):
        """Test lazy initialization of dependencies."""
        from big_mood_detector.core.dependencies import Container, Lazy
        
        # Track initialization
        initialized = []
        
        class ExpensiveService:
            def __init__(self):
                initialized.append(True)
                self.value = "expensive"
        
        container = Container()
        container.register_singleton(ExpensiveService)
        
        # Get lazy wrapper
        lazy_service = container.resolve(Lazy[ExpensiveService])
        
        # Should not be initialized yet
        assert len(initialized) == 0
        
        # Access value
        value = lazy_service.value.value
        
        # Now should be initialized
        assert len(initialized) == 1
        assert value == "expensive"

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        from big_mood_detector.core.dependencies import Container, CircularDependencyError
        
        container = Container()
        
        # Test direct circular detection by manually adding to resolving set
        # This tests the mechanism without creating an actual infinite loop
        key = container._get_key(str, None)
        container._resolving.add(key)
        
        # Should detect circular dependency
        with pytest.raises(CircularDependencyError):
            container.resolve(str)

    def test_named_dependencies(self):
        """Test named dependencies for multiple implementations."""
        from big_mood_detector.core.dependencies import Container
        
        # Create interface and implementations
        class Database:
            def connect(self): ...
        
        class PostgresDB(Database):
            def connect(self):
                return "postgres"
        
        class MongoDB(Database):
            def connect(self):
                return "mongo"
        
        container = Container()
        
        # Register named implementations
        container.register_singleton(Database, PostgresDB, name="postgres")
        container.register_singleton(Database, MongoDB, name="mongo")
        
        # Resolve by name
        postgres = container.resolve(Database, name="postgres")
        mongo = container.resolve(Database, name="mongo")
        
        assert postgres.connect() == "postgres"
        assert mongo.connect() == "mongo"

    def test_application_wide_container(self):
        """Test application-wide container setup."""
        from big_mood_detector.core.dependencies import setup_dependencies
        from big_mood_detector.core.config import Settings
        
        # Setup application dependencies
        container = setup_dependencies(Settings())
        
        # Verify core services are registered
        from big_mood_detector.application.use_cases.process_health_data_use_case import (
            MoodPredictionPipeline
        )
        from big_mood_detector.application.services.data_parsing_service import (
            DataParsingService
        )
        
        # Should be able to resolve core services
        pipeline = container.resolve(MoodPredictionPipeline)
        parser = container.resolve(DataParsingService)
        
        assert pipeline is not None
        assert parser is not None