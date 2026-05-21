#pragma once

#include "common/ducklake_data_file.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {
class DuckLakeTableEntry;

struct DuckLakePartitionFilterValue {
	bool is_null = false;
	string value;
};

struct DuckLakePartitionFilterEntry {
	idx_t partition_key_index;
	DuckLakePartitionFilterValue value;
};

class DuckLakePartitionFilter {
public:
	static DuckLakePartitionFilter Parse(DuckLakeTableEntry &table, const Value &filter);

	bool Matches(optional_idx file_partition_id, const vector<DuckLakeFilePartition> &file_values) const;

private:
	optional_idx partition_id;
	vector<DuckLakePartitionFilterEntry> values;
};

} // namespace duckdb
