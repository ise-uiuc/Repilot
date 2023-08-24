# Repilot Patch

```
            if (_kind == STD_LOCALE) {
                return Locale.ROOT;
            }
```

# Developer Patch

```
            if (_kind == STD_LOCALE) {
                return Locale.ROOT;
            }
```

# Context

```
--- bug/JacksonDatabind-42/src/main/java/com/fasterxml/jackson/databind/deser/std/FromStringDeserializer.java

+++ fix/JacksonDatabind-42/src/main/java/com/fasterxml/jackson/databind/deser/std/FromStringDeserializer.java

@@ -281,6 +281,9 @@

                 return URI.create("");
             }
             // As per [databind#1123], Locale too
+            if (_kind == STD_LOCALE) {
+                return Locale.ROOT;
+            }
             return super._deserializeFromEmptyString();
         }
     }
```

# Note

