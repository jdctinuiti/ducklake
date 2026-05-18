#include "functions/ducklake_table_functions.hpp"
#include "common/ducklake_data_file.hpp"
#include "storage/ducklake_metadata_manager.hpp"
#include "storage/ducklake_partition_data.hpp"
#include "storage/ducklake_table_entry.hpp"
#include "storage/ducklake_transaction.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

struct DropPartitionValue {
	bool is_null = false;
	string value;
};

struct DropPartitionKey {
	idx_t partition_key_index;
	LogicalType type;
};

struct DropDataFile {
	DataFileIndex file_id;
	string path;
	idx_t row_count = 0;
	vector<DropPartitionValue> partition_values;
};

struct DuckLakeDropDataFilesData : public TableFunctionData {
	DuckLakeDropDataFilesData(Catalog &catalog, DuckLakeTableEntry &table) : catalog(catalog), table(table) {
	}

	Catalog &catalog;
	DuckLakeTableEntry &table;
	unordered_map<idx_t, DropPartitionValue> partition_filter;
	bool dry_run = false;
};

struct DuckLakeDropDataFilesState : public GlobalTableFunctionState {
	idx_t offset = 0;
	bool finished = false;
	vector<vector<Value>> rows;
};

static unordered_map<string, DropPartitionKey> GetPartitionKeyMap(DuckLakeTableEntry &table) {
	auto partition_data = table.GetPartitionData();
	if (!partition_data) {
		throw InvalidInputException("ducklake_drop_data_files requires a partitioned table");
	}
	auto partition_sql_exprs = table.GetPartitionSQLExpressions();
	unordered_map<string, DropPartitionKey> result;
	case_insensitive_set_t used_names;
	for (auto &field : partition_data->fields) {
		auto &column = table.GetColumnByFieldId(field.field_id);
		auto field_name = column.GetName();
		auto partition_key_name =
		    DuckLakePartitionUtils::GetPartitionKeyName(field.transform.type, field_name, used_names);
		used_names.insert(partition_key_name);

		DropPartitionKey key {field.partition_key_index,
		                      DuckLakePartitionUtils::GetPartitionKeyType(field.transform.type, column.GetType())};
		result[StringUtil::Lower(partition_key_name)] = key;
		result[StringUtil::Lower(field_name)] = key;
		if (field.partition_key_index < partition_sql_exprs.size()) {
			result[StringUtil::Lower(partition_sql_exprs[field.partition_key_index])] = key;
		}
	}
	return result;
}

static DropPartitionValue CastPartitionValue(const Value &input_value, const LogicalType &type,
                                             const string &partition_key_name) {
	DropPartitionValue result;
	result.is_null = input_value.IsNull();
	if (result.is_null) {
		return result;
	}
	Value cast_value;
	string error;
	if (!input_value.DefaultTryCastAs(type, cast_value, &error)) {
		throw InvalidInputException("Failed to cast partition value for key \"%s\" to %s: %s", partition_key_name,
		                            type.ToString(), error);
	}
	result.value = cast_value.ToString();
	return result;
}

static unordered_map<idx_t, DropPartitionValue> ParsePartitionFilter(DuckLakeTableEntry &table, const Value &filter) {
	if (filter.IsNull()) {
		throw InvalidInputException("partition_values cannot be NULL");
	}
	if (filter.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("partition_values must be a STRUCT");
	}
	auto partition_keys = GetPartitionKeyMap(table);
	unordered_map<idx_t, DropPartitionValue> result;
	auto &children = StructValue::GetChildren(filter);
	for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
		auto partition_key_name = StructType::GetChildName(filter.type(), child_idx);
		auto key = StringUtil::Lower(partition_key_name);
		auto partition_key = partition_keys.find(key);
		if (partition_key == partition_keys.end()) {
			throw InvalidInputException("Unknown partition key \"%s\" for table \"%s\"", partition_key_name, table.name);
		}
		result[partition_key->second.partition_key_index] =
		    CastPartitionValue(children[child_idx], partition_key->second.type, partition_key_name);
	}
	if (result.empty()) {
		throw InvalidInputException("partition_values cannot be empty");
	}
	return result;
}

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
	result->partition_filter = ParsePartitionFilter(table, partition_values_entry->second);
	for (auto &entry : input.named_parameters) {
		auto lower = StringUtil::Lower(entry.first);
		if (lower == "dry_run") {
			result->dry_run = BooleanValue::Get(entry.second);
		} else if (lower != "schema" && lower != "partition_values") {
			throw InternalException("Unknown named parameter %s for drop_data_files", entry.first);
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

static bool PartitionValueMatches(const DropPartitionValue &expected, const DropPartitionValue &actual) {
	if (expected.is_null || actual.is_null) {
		return expected.is_null && actual.is_null;
	}
	return expected.value == actual.value;
}

static bool FileMatches(const DropDataFile &file, const unordered_map<idx_t, DropPartitionValue> &partition_filter) {
	for (auto &entry : partition_filter) {
		if (entry.first >= file.partition_values.size()) {
			return false;
		}
		if (!PartitionValueMatches(entry.second, file.partition_values[entry.first])) {
			return false;
		}
	}
	return true;
}

static DropPartitionValue ConvertPartitionValue(const Value &value) {
	DropPartitionValue result;
	result.is_null = value.IsNull();
	if (!result.is_null) {
		result.value = value.ToString();
	}
	return result;
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
SELECT data.data_file_id, data.record_count, part.partition_key_index, part.partition_value
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
	auto partition_data = table.GetPartitionData();
	for (auto &row : *rows) {
		auto data_file_id = row.GetValue<idx_t>(0);
		auto file_entry = result.find(data_file_id);
		if (file_entry == result.end()) {
			continue;
		}
		auto &file = file_entry->second;
		file.row_count = row.GetValue<idx_t>(1);
		if (file.partition_values.empty() && partition_data) {
			file.partition_values.resize(partition_data->fields.size());
			for (auto &partition_value : file.partition_values) {
				partition_value.is_null = true;
			}
		}
		if (row.IsNull(2)) {
			continue;
		}
		auto partition_key_index = row.GetValue<idx_t>(2);
		if (partition_key_index >= file.partition_values.size()) {
			throw InternalException("Invalid DuckLake partition key index");
		}
		auto &partition_value = file.partition_values[partition_key_index];
		partition_value.is_null = row.IsNull(3);
		if (!partition_value.is_null) {
			partition_value.value = row.GetValue<string>(3);
		}
	}
	return result;
}

static vector<DropDataFile> GetTransactionLocalDropDataFiles(DuckLakeTransaction &transaction, DuckLakeTableEntry &table) {
	vector<DropDataFile> result;
	auto transaction_local_files = transaction.GetTransactionLocalFiles(table.GetTableId());
	for (auto &file : transaction_local_files) {
		DropDataFile drop_file;
		drop_file.path = file.file_name;
		drop_file.row_count = file.row_count;
		auto partition_data = table.GetPartitionData();
		if (partition_data) {
			drop_file.partition_values.resize(partition_data->fields.size());
			for (auto &partition_value : drop_file.partition_values) {
				partition_value.is_null = true;
			}
		}
		for (auto &partition : file.partition_values) {
			if (partition.partition_column_idx >= drop_file.partition_values.size()) {
				throw InternalException("Invalid DuckLake transaction-local partition key index");
			}
			drop_file.partition_values[partition.partition_column_idx] = ConvertPartitionValue(partition.partition_value);
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
			if (!FileMatches(file, bind_data.partition_filter)) {
				continue;
			}
			if (!bind_data.dry_run) {
				transaction.DropFile(bind_data.table.GetTableId(), file.file_id, file.path);
			}
			state.rows.push_back({Value(file.path), Value::UBIGINT(file.row_count)});
		}
		auto transaction_local_files = GetTransactionLocalDropDataFiles(transaction, bind_data.table);
		for (auto &file : transaction_local_files) {
			if (!FileMatches(file, bind_data.partition_filter)) {
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
