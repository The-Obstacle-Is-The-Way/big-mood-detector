# Labels API

## GET /api/v1/labels

Retrieve all mood episode labels.

### Query Parameters

- `user_id` (optional) - Filter by user ID
- `start_date` (optional) - Filter labels after this date
- `end_date` (optional) - Filter labels before this date
- `episode_type` (optional) - Filter by episode type (depression, hypomanic, manic)

### Response

```json
{
  "labels": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user123",
      "start_date": "2024-01-10",
      "end_date": "2024-01-17",
      "episode_type": "depression",
      "severity": "moderate",
      "confidence": 0.85,
      "source": "clinician",
      "created_at": "2024-01-18T10:30:00Z"
    }
  ],
  "total_count": 42,
  "page": 1,
  "page_size": 20
}
```

## POST /api/v1/labels

Create a new mood episode label.

### Request Body

```json
{
  "user_id": "user123",
  "start_date": "2024-01-10",
  "end_date": "2024-01-17",
  "episode_type": "depression",
  "severity": "moderate",
  "confidence": 0.85,
  "notes": "Patient reported low mood and fatigue"
}
```

### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Label created successfully"
}
```

## PUT /api/v1/labels/{label_id}

Update an existing label.

## DELETE /api/v1/labels/{label_id}

Delete a label.

### Status Codes

- `200 OK` - Success
- `201 Created` - Label created
- `404 Not Found` - Label not found
- `422 Unprocessable Entity` - Validation failed