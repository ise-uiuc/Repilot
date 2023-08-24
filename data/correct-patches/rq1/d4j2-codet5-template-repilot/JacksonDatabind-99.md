# Repilot Patch

```

        sb.append('>');
```

# Developer Patch

```
        sb.append('>');
```

# Context

```
--- bug/JacksonDatabind-99/src/main/java/com/fasterxml/jackson/databind/type/ReferenceType.java

+++ fix/JacksonDatabind-99/src/main/java/com/fasterxml/jackson/databind/type/ReferenceType.java

@@ -166,6 +166,8 @@

         sb.append(_class.getName());
         sb.append('<');
         sb.append(_referencedType.toCanonical());
+
+        sb.append('>');
         return sb.toString();
     }
```

# Note

