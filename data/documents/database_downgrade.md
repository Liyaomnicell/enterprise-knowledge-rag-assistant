# Database Downgrade Troubleshooting Guide

## Problem

A software deployment may fail when downgrading a database from a newer application version to an older version.

## Symptoms

Typical symptoms include:

- Deployment script exits with an error.
- Database schema deployment reports possible data loss.
- Columns or tables introduced by the newer version cannot be mapped to the older schema.
- Application startup fails after downgrade.

## Root Cause

Database downgrade may require destructive schema operations such as:

- Dropping columns
- Dropping tables
- Changing data types
- Removing indexes

These changes can result in data loss.

Setting a deployment option to allow possible data loss only disables the pre-deployment blocking check. It does not guarantee that the downgrade is safe.

## Recommended Investigation

1. Generate a deployment report before modifying the database.
2. Review DROP and ALTER operations.
3. Back up the target database.
4. Verify whether data migration is required.
5. Run the downgrade in a test environment.
6. Validate application compatibility after the downgrade.

## Recommendation

Database downgrade should never be treated as a simple reverse upgrade. A downgrade strategy must explicitly define how schema and data differences are handled.
