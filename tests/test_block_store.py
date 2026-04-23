from pathlib import Path

from amlx.cache.blocks import PagedBlockStore


def test_paged_block_store_roundtrip() -> None:
    base = Path('.cache/block-test')
    store = PagedBlockStore(
        root_dir=base / 'blocks',
        index_db=base / 'blocks.sqlite3',
        block_chars=10,
    )

    value = 'abcdefghijklmnopqrstuvwxyz'
    count = store.put(cache_key='k1', value=value)
    assert count >= 3

    loaded = store.get(cache_key='k1')
    assert loaded == value
