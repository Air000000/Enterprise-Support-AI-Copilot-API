from experiments.rag_local.query_chroma import build_where_filter


def test_build_where_filter_returns_none_without_filters():
    assert build_where_filter() is None


def test_build_where_filter_with_tenant_id_only():
    assert build_where_filter(tenant_id="tenant_demo") == {
        "tenant_id": "tenant_demo"
    }


def test_build_where_filter_with_category_only():
    assert build_where_filter(category="it") == {
        "category": "it"
    }


def test_build_where_filter_with_tenant_id_and_category():
    assert build_where_filter(
        tenant_id="tenant_demo",
        category="it",
    ) == {
        "$and": [
            {"tenant_id": "tenant_demo"},
            {"category": "it"},
        ]
    }