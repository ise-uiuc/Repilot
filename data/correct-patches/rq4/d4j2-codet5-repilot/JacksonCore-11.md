# Repilot Patch

```

            _verifyNeedForRehash();
```

# Developer Patch

```
            _verifyNeedForRehash();
```

# Context

```
--- bug/JacksonCore-11/src/main/java/com/fasterxml/jackson/core/sym/ByteQuadsCanonicalizer.java

+++ fix/JacksonCore-11/src/main/java/com/fasterxml/jackson/core/sym/ByteQuadsCanonicalizer.java

@@ -879,6 +879,8 @@

             _hashShared = false;
             // 09-Sep-2015, tatu: As per [jackson-core#216], also need to ensure
             //    we rehash as needed, as need-rehash flag is not copied from parent
+
+            _verifyNeedForRehash();
         }
         if (_needRehash) {
             rehash();
```

# Note

