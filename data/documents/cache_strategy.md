# Application Cache Strategy

Caching can reduce database load and improve application response time.

## Suitable Data

Caching is appropriate for:

- Frequently read data
- Expensive computations
- Data that changes infrequently

## Risks

Common caching risks include:

- Stale data
- Cache stampede
- Inconsistent cache invalidation
- Memory pressure

## Recommended Pattern

Use cache-aside when the application can tolerate temporary cache misses.

The application first checks the cache. If the value is missing, it retrieves the value from the database and stores it in the cache.

TTL values should be selected based on business requirements and acceptable staleness.

