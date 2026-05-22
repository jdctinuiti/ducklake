#include "functions/ducklake_table_functions.hpp"
#include "common/ducklake_data_file.hpp"
#include "storage/ducklake_metadata_manager.hpp"
#include "storage/ducklake_partition_filter.hpp"
#include "storage/ducklake_table_entry.hpp"
#include "storage/ducklake_transaction.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

struct DropDataFile {
	DataFileIndex file_id;
	string path;
	idx_t row_count = 0;
	optional_idx partition_id;
	vector<DuckLakeFilePartition> partition_values;
};

struct DuckLakeDropDataFilesData : public TableFunctionData {
	DuckLakeDropDataFilesData(Catalog &catalog, DuckLakeTableEntry &table) : catalog(catalog), table(table) {
	}

	Catalog &catalog;
	DuckLakeTableEntry &table;
	DuckLakePartitionFilter partition_filter;
	bool dry_run = false;
};

struct DuckLakeDropDataFilesState : public GlobalTableFunctionState {
	idx_t offset = 0;
	bool finished = false;
	vector<vector<Value>> rows;
};

static unique_ptr<FunctionData> DuckLakeDropDataFilesBind(ClientContext &context, TableFunctionBindInput &input,
                                                          vector<LogicalType> &return_types, vector<string> &names) {
	auto &catalog = DuckLakeBaseMetadataFunction::GetCatalog(context, input.inputs[0]);
	string schema_name;
	if (input.inputs[1].IsNull()) {
		throw InvalidInputException("Table name cannot be NULL");
	}
	if (input.named_parameters.find("schema") != input.named_parameters.end()) {
		schema_name = StringValue::Get(input.named_parameters["schema"]);
	}
	auto table_name = StringValue::Get(input.inputs[1]);
	auto entry =
	    catalog.GetEntry<TableCatalogEntry>(context, schema_name, table_name, OnEntryNotFound::THROW_EXCEPTION);
	auto &table = entry->Cast<DuckLakeTableEntry>();

	auto result = make_uniq<DuckLakeDropDataFilesData>(catalog, table);
	auto partition_values_entry = input.named_parameters.find("partition_values");
	if (partition_values_entry == input.named_parameters.end()) {
		throw InvalidInputException("partition_values is required");
	}
	result->partition_filter = DuckLakePartitionFilter::Parse(table, partition_values_entry->second);
	for (auto &entry : input.named_parameters) {
		auto lower = StringUtil::Lower(entry.first);
		if (lower == "dry_run") {
			result->dry_run = BooleanValue::Get(entry.second);
		} else if (lower != "schema" && lower != "partition_values") {
			throw InvalidInputException("Unknown named parameter %s for drop_data_files", entry.first);
		}
	}

	names.emplace_back("filename");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("row_count");
	return_types.emplace_back(LogicalType::UBIGINT);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DuckLakeDropDataFilesInit(ClientContext &context,
                                                                      TableFunctionInitInput &input) {
	return make_uniq<DuckLakeDropDataFilesState>();
}

static unordered_map<idx_t, DropDataFile> GetDropDataFiles(DuckLakeTransaction &transaction,
                                                           DuckLakeTableEntry &table) {
	auto &metadata_manager = transaction.GetMetadataManager();
	auto active_files = metadata_manager.GetFilesForTable(table, transaction.GetSnapshot(), nullptr);
	unordered_map<idx_t, DropDataFile> result;
	vector<string> file_ids;
	for (auto &file : active_files) {
		if (file.data_type != DuckLakeDataType::DATA_FILE || !file.file_id.IsValid()) {
			continue;
		}
		if (transaction.FileIsDropped(file.file.path)) {
			continue;
		}
		DropDataFile drop_file;
		drop_file.file_id = file.file_id;
		drop_file.path = file.file.path;
		result[file.file_id.index] = std::move(drop_file);
		file_ids.push_back(std::to_string(file.file_id.index));
	}
	if (result.empty()) {
		return result;
	}

	auto query = StringUtil::Format(R"(
SELECT data.data_file_id, data.record_count, data.partition_id, part.partition_key_index, part.partition_value
FROM {METADATA_CATALOG}.ducklake_data_file data
LEFT JOIN {METADATA_CATALOG}.ducklake_file_partition_value part
  ON data.data_file_id = part.data_file_id AND data.table_id = part.table_id
WHERE data.table_id=%d
  AND data.data_file_id IN (%s)
ORDER BY data.data_file_id, part.partition_key_index
)",
	                                table.GetTableId().index, StringUtil::Join(file_ids, ", "));
	auto rows = transaction.Query(query);
	if (rows->HasError()) {
		rows->GetErrorObject().Throw("Failed to get DuckLake partition values: ");
	}
	for (auto &row : *rows) {
		auto data_file_id = row.GetValue<idx_t>(0);
		auto file_entry = result.find(data_file_id);
		if (file_entry == result.end()) {
			continue;
		}
		auto &file = file_entry->second;
		file.row_count = row.GetValue<idx_t>(1);
		file.partition_id = row.IsNull(2) ? optional_idx() : row.GetValue<idx_t>(2);
		if (row.IsNull(3)) {
			continue;
		}
		DuckLakeFilePartition partition_value;
		partition_value.partition_column_idx = row.GetValue<idx_t>(3);
		partition_value.partition_value = row.IsNull(4) ? Value(LogicalType::VARCHAR) : Value(row.GetValue<string>(4));
		file.partition_values.push_back(std::move(partition_value));
	}
	return result;
}

static vector<DropDataFile> GetTransactionLocalDropDataFiles(DuckLakeTransaction &transaction, TableIndex table_id) {
	vector<DropDataFile> result;
	auto transaction_local_files = transaction.GetTransactionLocalFiles(table_id);
	for (auto &file : transaction_local_files) {
		DropDataFile drop_file;
		drop_file.path = file.file_name;
		drop_file.row_count = file.row_count;
		for (auto &partition : file.partition_values) {
			drop_file.partition_values.push_back(partition);
		}
		result.push_back(std::move(drop_file));
	}
	return result;
}

static void DuckLakeDropDataFilesExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<DuckLakeDropDataFilesState>();
	if (!state.finished) {
		auto &bind_data = data_p.bind_data->Cast<DuckLakeDropDataFilesData>();
		auto &transaction = DuckLakeTransaction::Get(context, bind_data.catalog);
		auto files = GetDropDataFiles(transaction, bind_data.table);
		for (auto &entry : files) {
			auto &file = entry.second;
			if (!bind_data.partition_filter.Matches(file.partition_id, file.partition_values)) {
				continue;
			}
			if (!bind_data.dry_run) {
				transaction.DropFile(bind_data.table.GetTableId(), file.file_id, file.path);
			}
			state.rows.push_back({Value(file.path), Value::UBIGINT(file.row_count)});
		}
		auto transaction_local_files = GetTransactionLocalDropDataFiles(transaction, bind_data.table.GetTableId());
		for (auto &file : transaction_local_files) {
			if (!bind_data.partition_filter.Matches(file.partition_id, file.partition_values)) {
				continue;
			}
			if (!bind_data.dry_run) {
				transaction.DropTransactionLocalFile(bind_data.table.GetTableId(), file.path);
			}
			state.rows.push_back({Value(file.path), Value::UBIGINT(file.row_count)});
		}
		state.finished = true;
	}

	idx_t count = 0;
	while (state.offset < state.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = state.rows[state.offset++];
		output.data[0].Append(entry[0]);
		output.data[1].Append(entry[1]);
		count++;
	}
	output.SetChildCardinality(count);
}

TableFunctionSet DuckLakeDropDataFilesFunction::GetFunctions() {
	TableFunctionSet set("ducklake_drop_data_files");
	TableFunction function("ducklake_drop_data_files", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                       DuckLakeDropDataFilesExecute, DuckLakeDropDataFilesBind, DuckLakeDropDataFilesInit);
	function.named_parameters["schema"] = LogicalType::VARCHAR;
	function.named_parameters["partition_values"] = LogicalType::ANY;
	function.named_parameters["dry_run"] = LogicalType::BOOLEAN;
	set.AddFunction(function);
	return set;
}

} // namespace duckdb
